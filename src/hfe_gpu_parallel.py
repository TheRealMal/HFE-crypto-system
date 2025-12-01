"""
GPU-параллельная реализация HFE с использованием PyTorch для CUDA
"""
import torch
from typing import Optional
import logging
from .hfe_base import HFEBase
from .field_operations import GF2n

logger = logging.getLogger(__name__)

# Проверка доступности CUDA
try:
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception as e:
    CUDA_AVAILABLE = False
    logger.warning(f"Ошибка при проверке CUDA: {e}")


class GF2nGPU:
    """
    Класс для работы с конечным полем GF(2^n) на GPU с использованием PyTorch
    Оптимизирован для пакетной обработки
    """
    
    def __init__(self, n: int, device: torch.device):
        """
        Инициализация поля GF(2^n) на GPU
        
        Args:
            n: Размерность поля
            device: Устройство PyTorch (cuda или cpu)
        """
        self.n = n
        self.device = device
        self.field_size = 2 ** n
        
        # Получаем неприводимый многочлен из CPU-версии
        cpu_field = GF2n(n)
        self.irreducible = cpu_field.irreducible
        
        # Преобразуем в тензор для GPU
        self.irreducible_tensor = torch.tensor(
            self.irreducible, dtype=torch.int64, device=device
        )
        
        logger.debug(f"GF2nGPU инициализирован: n={n}, device={device}")
    
    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Сложение в GF(2^n) - XOR операция
        
        Args:
            a, b: Тензоры элементов поля
            
        Returns:
            Тензор суммы
        """
        return a ^ b
    
    def multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Умножение в GF(2^n) с приведением по модулю неприводимого многочлена
        Оптимизировано для пакетной обработки
        
        Args:
            a, b: Тензоры элементов поля (могут быть батчами)
            
        Returns:
            Тензор произведения
        """
        # Нулевые элементы
        zero_mask = (a == 0) | (b == 0)
        result = torch.zeros_like(a)
        
        # Полиномиальное умножение с приведением по модулю
        # Для каждого бита b вычисляем a << i и суммируем (XOR)
        for i in range(self.n):
            bit_mask = (b >> i) & 1
            shifted = a << i
            result = result ^ (shifted * bit_mask)
        
        # Приведение по модулю неприводимого многочлена
        result = self._reduce_mod_polynomial(result)
        
        # Устанавливаем нули там, где нужно
        result[zero_mask] = 0
        
        return result
    
    def _reduce_mod_polynomial(self, value: torch.Tensor) -> torch.Tensor:
        """
        Приведение многочлена по модулю неприводимого многочлена
        Использует итеративное приведение старших битов
        
        Args:
            value: Тензор значений для приведения
            
        Returns:
            Приведенные значения
        """
        result = value.clone()
        irreducible = self.irreducible_tensor.item()
        mask = (1 << self.n) - 1
        max_degree = 2 * self.n - 1  # Максимальная степень после умножения
        
        # Итеративно приводим, начиная со старших битов
        for bit_pos in range(max_degree, self.n - 1, -1):
            # Находим элементы с установленным битом на позиции bit_pos
            bit_mask = (result >> bit_pos) & 1
            
            if bit_mask.any():
                # Вычисляем, на сколько нужно сдвинуть неприводимый многочлен
                shift = bit_pos - self.n
                # Применяем приведение: XOR с сдвинутым неприводимым многочленом
                reduced = (irreducible << shift) & ((1 << (bit_pos + 1)) - 1)
                result = result ^ (reduced * bit_mask)
        
        return result & mask
    
    def power(self, a: torch.Tensor, exp: int) -> torch.Tensor:
        """
        Возведение в степень в GF(2^n) методом быстрого возведения
        
        Args:
            a: Тензор баз
            exp: Показатель степени
            
        Returns:
            Тензор результатов
        """
        if exp == 0:
            return torch.ones_like(a)
        if exp == 1:
            return a
        
        result = torch.ones_like(a)
        base = a.clone()
        
        while exp > 0:
            if exp & 1:
                result = self.multiply(result, base)
            base = self.multiply(base, base)
            exp >>= 1
        
        return result
    
    def vector_to_field(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Преобразование векторов бит в элементы поля
        
        Args:
            vectors: Тензор формы (batch_size, n) или (n,)
            
        Returns:
            Тензор элементов поля формы (batch_size,) или скаляр
        """
        # Создаем степени 2^i для i от 0 до n-1
        powers = torch.arange(self.n, device=self.device, dtype=torch.int64)
        powers_2 = (1 << powers)  # [1, 2, 4, 8, ...]
        
        if vectors.dim() == 1:
            # Один вектор: (n,) -> скаляр
            return (vectors.long() * powers_2).sum()
        else:
            # Батч векторов: (batch_size, n) -> (batch_size,)
            return (vectors.long() * powers_2).sum(dim=-1)
    
    def field_to_vector(self, elements: torch.Tensor) -> torch.Tensor:
        """
        Преобразование элементов поля в векторы бит
        
        Args:
            elements: Тензор элементов поля формы (batch_size,) или скаляр
            
        Returns:
            Тензор векторов формы (batch_size, n) или (n,)
        """
        # Создаем маску для извлечения битов
        powers = torch.arange(self.n, device=self.device, dtype=torch.int64)
        
        if elements.dim() == 0:
            # Скаляр -> (n,)
            return ((elements.unsqueeze(-1) >> powers) & 1).long()
        else:
            # Батч: (batch_size,) -> (batch_size, n)
            return ((elements.unsqueeze(-1) >> powers) & 1).long()


class HFEGPUParallel(HFEBase):
    """
    GPU-параллельная реализация HFE с использованием PyTorch
    Оптимизирована для пакетной обработки больших объемов данных
    """
    
    def __init__(self, n: int = 8, d: int = 3, seed: Optional[int] = None,
                 device: Optional[str] = None, batch_size: int = 1024):
        """
        Инициализация GPU-параллельного HFE
        
        Args:
            n: Размерность поля (GF(2^n))
            d: Степень многочлена HFE
            seed: Seed для генерации случайных ключей
            device: Устройство ('cuda', 'cpu' или None для автоопределения)
            batch_size: Размер батча для обработки (рекомендуется >= 1024)
        """
        if not CUDA_AVAILABLE and device != 'cpu':
            raise RuntimeError(
                "CUDA недоступен. Установите PyTorch с поддержкой CUDA или используйте device='cpu'"
            )
        
        super().__init__(n, d, seed)
        
        # Определение устройства
        if device is None:
            self.device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.batch_size = batch_size
        
        # Инициализация поля на GPU
        self.gpu_field = GF2nGPU(n, self.device)
        
        # Перенос ключей на GPU
        self._transfer_keys_to_gpu()
        
        logger.info(
            f"Инициализация GPU-параллельного HFE: n={n}, d={d}, "
            f"device={self.device}, batch_size={batch_size}"
        )
    
    def _transfer_keys_to_gpu(self):
        """Перенос секретных ключей на GPU в виде тензоров"""
        # Преобразуем матрицы и векторы в тензоры PyTorch
        # Используем float32 для матричных операций (CUDA не поддерживает Long для matmul)
        # Для битовых операций будем преобразовывать обратно в int64
        self.S1_gpu = torch.tensor(
            self.S1, dtype=torch.float32, device=self.device
        )
        self.S0_gpu = torch.tensor(
            self.S0, dtype=torch.float32, device=self.device
        )
        self.T1_gpu = torch.tensor(
            self.T1, dtype=torch.float32, device=self.device
        )
        self.T0_gpu = torch.tensor(
            self.T0, dtype=torch.float32, device=self.device
        )
        self.S1_inv_gpu = torch.tensor(
            self.S1_inv, dtype=torch.float32, device=self.device
        )
        self.T1_inv_gpu = torch.tensor(
            self.T1_inv, dtype=torch.float32, device=self.device
        )
        
        logger.debug("Ключи перенесены на GPU")
    
    def _affine_transform_gpu(self, x: torch.Tensor, A: torch.Tensor, 
                               b: torch.Tensor) -> torch.Tensor:
        """
        Применение аффинного преобразования на GPU: y = A*x + b (mod 2)
        
        Args:
            x: Тензор векторов формы (batch_size, n) или (n,), dtype int64
            A: Матрица преобразования (n, n), dtype float32
            b: Вектор смещения (n,), dtype float32
            
        Returns:
            Тензор результатов формы (batch_size, n) или (n,), dtype int64
        """
        # Преобразуем вход в float32 для матричных операций
        x_float = x.float()
        
        if x.dim() == 1:
            # Один вектор: (n,) -> (n,)
            # y = A @ x + b
            y = (A @ x_float + b) % 2
        else:
            # Батч векторов: (batch_size, n) -> (batch_size, n)
            # y[i] = A @ x[i] + b для каждого i
            # Можно записать как: y = x @ A.T + b
            y = (x_float @ A.T + b.unsqueeze(0)) % 2
        
        # Преобразуем обратно в int64 и берем по модулю 2
        return (y.long() % 2).long()
    
    def _hfe_polynomial_gpu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вычисление значения HFE многочлена на GPU
        P(x) = sum(x^(2^i)) для i от 1 до d
        
        Args:
            x: Тензор элементов поля (batch_size,) или скаляр
            
        Returns:
            Тензор результатов
        """
        result = torch.zeros_like(x)
        
        for i in range(1, self.d + 1):
            power = 2 ** i
            if power < self.field_size:
                # Вычисляем x^(2^i) используя свойства поля характеристики 2
                # В GF(2^n) возведение в степень 2^i - это линейная операция (автоморфизм Фробениуса)
                # Для упрощения используем обычное возведение в степень
                powered = self.gpu_field.power(x, power)
                result = self.gpu_field.add(result, powered)
        
        return result
    
    def _solve_hfe_gpu(self, y: torch.Tensor) -> torch.Tensor:
        """
        Решение уравнения HFE на GPU: найти x такой, что P(x) = y
        Использует параллельный перебор для батча значений
        
        Args:
            y: Тензор значений (batch_size,) или скаляр
            
        Returns:
            Тензор решений
        """
        if y.dim() == 0:
            # Скаляр - используем CPU версию для простоты
            y_cpu = y.item()
            x_cpu = self._solve_hfe(y_cpu)
            return torch.tensor(x_cpu, dtype=torch.int64, device=self.device)
        
        # Батч: создаем сетку всех возможных значений x
        batch_size = y.shape[0]
        
        # Для больших полей это может быть неэффективно, но для малых (n<=8) работает
        if self.field_size > 1024:
            # Для больших полей используем последовательный перебор на CPU
            logger.warning(
                f"Поле слишком большое (field_size={self.field_size}), "
                "используется последовательный перебор"
            )
            solutions = []
            y_cpu = y.cpu().numpy()
            for y_val in y_cpu:
                x_cpu = self._solve_hfe(int(y_val))
                solutions.append(x_cpu)
            return torch.tensor(solutions, dtype=torch.int64, device=self.device)
        
        # Параллельный перебор на GPU
        all_x = torch.arange(
            self.field_size, dtype=torch.int64, device=self.device
        ).unsqueeze(0).expand(batch_size, -1)  # (batch_size, field_size)
        
        # Вычисляем P(x) для всех возможных x
        all_x_flat = all_x.flatten()  # (batch_size * field_size,)
        all_px = self._hfe_polynomial_gpu(all_x_flat)  # (batch_size * field_size,)
        all_px = all_px.view(batch_size, self.field_size)  # (batch_size, field_size)
        
        # Расширяем y для сравнения
        y_expanded = y.unsqueeze(-1).expand(-1, self.field_size)  # (batch_size, field_size)
        
        # Находим совпадения
        matches = (all_px == y_expanded)  # (batch_size, field_size)
        
        # Берем первое совпадение для каждого элемента батча
        # Если совпадений нет, возвращаем 0
        match_indices = matches.long().argmax(dim=1)  # (batch_size,)
        batch_indices = torch.arange(batch_size, device=self.device)
        solutions = all_x[batch_indices, match_indices]
        
        # Проверяем, что решение действительно правильное
        # (если argmax вернул 0, но это не решение, значит решения нет)
        px_solutions = self._hfe_polynomial_gpu(solutions)
        valid = (px_solutions == y)
        solutions[~valid] = 0
        
        return solutions
    
    def encrypt_batch(self, plaintexts: torch.Tensor) -> torch.Tensor:
        """
        Пакетное шифрование на GPU
        
        Args:
            plaintexts: Тензор открытых текстов формы (batch_size, n)
            
        Returns:
            Тензор зашифрованных текстов формы (batch_size, n)
        """
        # Шаг 1: Применение S
        x = self._affine_transform_gpu(plaintexts, self.S1_gpu, self.S0_gpu)
        x_field = self.gpu_field.vector_to_field(x)
        
        # Шаг 2: Применение HFE многочлена
        y_field = self._hfe_polynomial_gpu(x_field)
        y = self.gpu_field.field_to_vector(y_field)
        
        # Шаг 3: Применение T
        ciphertexts = self._affine_transform_gpu(y, self.T1_gpu, self.T0_gpu)
        
        return ciphertexts
    
    def decrypt_batch(self, ciphertexts: torch.Tensor) -> torch.Tensor:
        """
        Пакетное расшифрование на GPU
        
        Args:
            ciphertexts: Тензор зашифрованных текстов формы (batch_size, n)
            
        Returns:
            Тензор расшифрованных текстов формы (batch_size, n)
        """
        # Шаг 1: Обратное преобразование T
        T0_neg = (-self.T0_gpu) % 2
        T1_inv_T0 = (self.T1_inv_gpu @ T0_neg) % 2
        y = self._affine_transform_gpu(ciphertexts, self.T1_inv_gpu, T1_inv_T0)
        y_field = self.gpu_field.vector_to_field(y)
        
        # Шаг 2: Решение уравнения HFE
        x_field = self._solve_hfe_gpu(y_field)
        x = self.gpu_field.field_to_vector(x_field)
        
        # Шаг 3: Обратное преобразование S
        S0_neg = (-self.S0_gpu) % 2
        S1_inv_S0 = (self.S1_inv_gpu @ S0_neg) % 2
        plaintexts = self._affine_transform_gpu(x, self.S1_inv_gpu, S1_inv_S0)
        
        return plaintexts
    
    def encrypt_block(self, data: bytes) -> bytes:
        """
        Шифрование блока данных на GPU с пакетной обработкой
        
        Args:
            data: Байты для шифрования
            
        Returns:
            Зашифрованные байты
        """
        logger.info(f"GPU-параллельное шифрование блока данных размером {len(data)} байт")
        
        # Преобразование байтов в векторы бит
        plaintexts_list = [
            [(byte_val >> i) & 1 for i in range(self.n)]
            for byte_val in data
        ]
        
        # Обработка батчами
        result_bytes = []
        
        for i in range(0, len(plaintexts_list), self.batch_size):
            batch = plaintexts_list[i:i + self.batch_size]
            
            # Преобразуем в тензор
            plaintexts_tensor = torch.tensor(
                batch, dtype=torch.int64, device=self.device
            )
            
            # Шифруем на GPU
            ciphertexts_tensor = self.encrypt_batch(plaintexts_tensor)
            
            # Преобразуем обратно в байты
            ciphertexts_cpu = ciphertexts_tensor.cpu().numpy()
            for ciphertext in ciphertexts_cpu:
                cipher_byte = sum(int(bit) << j for j, bit in enumerate(ciphertext))
                result_bytes.append(cipher_byte)
        
        logger.info(f"GPU-параллельное шифрование завершено, получено {len(result_bytes)} байт")
        return bytes(result_bytes)
    
    def decrypt_block(self, data: bytes) -> bytes:
        """
        Расшифрование блока данных на GPU с пакетной обработкой
        
        Args:
            data: Зашифрованные байты
            
        Returns:
            Расшифрованные байты
        """
        logger.info(f"GPU-параллельное расшифрование блока данных размером {len(data)} байт")
        
        # Преобразование байтов в векторы бит
        ciphertexts_list = [
            [(byte_val >> i) & 1 for i in range(self.n)]
            for byte_val in data
        ]
        
        # Обработка батчами
        result_bytes = []
        
        for i in range(0, len(ciphertexts_list), self.batch_size):
            batch = ciphertexts_list[i:i + self.batch_size]
            
            # Преобразуем в тензор
            ciphertexts_tensor = torch.tensor(
                batch, dtype=torch.int64, device=self.device
            )
            
            # Расшифровываем на GPU
            plaintexts_tensor = self.decrypt_batch(ciphertexts_tensor)
            
            # Преобразуем обратно в байты
            plaintexts_cpu = plaintexts_tensor.cpu().numpy()
            for plaintext in plaintexts_cpu:
                plain_byte = sum(int(bit) << j for j, bit in enumerate(plaintext))
                result_bytes.append(plain_byte)
        
        logger.info(f"GPU-параллельное расшифрование завершено, получено {len(result_bytes)} байт")
        return bytes(result_bytes)

