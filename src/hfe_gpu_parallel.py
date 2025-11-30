"""
GPU-параллельная реализация HFE с использованием numba CUDA
"""
import numpy as np
from typing import List, Tuple, Optional
import logging

try:
    from numba import cuda, types
    from numba.cuda import jit as cuda_jit
    _cuda_module = cuda  # Сохраняем модуль cuda
    # Попытка проверить доступность CUDA драйвера
    try:
        # Простая проверка доступности CUDA
        # Это может вызвать исключение, если драйвер не найден
        _cuda_available = cuda.is_available()
        if _cuda_available:
            CUDA_AVAILABLE = True
        else:
            CUDA_AVAILABLE = False
            cuda = None
            cuda_jit = lambda x: x
    except Exception as e:
        # CUDA установлен, но драйвер недоступен (например, на macOS без NVIDIA GPU)
        CUDA_AVAILABLE = False
        cuda = None
        cuda_jit = lambda x: x
        _cuda_error = str(e)
except ImportError:
    # Numba CUDA не установлен
    CUDA_AVAILABLE = False
    _cuda_module = None
    cuda = None
    cuda_jit = lambda x: x
    _cuda_error = "Numba CUDA не установлен"

# Инициализация логгера
logger = logging.getLogger(__name__)
if not CUDA_AVAILABLE:
    logger.debug("CUDA недоступен. GPU-параллелизация будет недоступна.")
    if '_cuda_error' in locals():
        logger.debug(f"Причина: {_cuda_error}")
    logger.debug("Это нормально на системах без NVIDIA GPU (например, macOS с Apple Silicon/Intel)")

from .hfe_base import HFEBase

# Определяем device functions только если CUDA доступен
if CUDA_AVAILABLE and '_cuda_module' in locals() and _cuda_module is not None:
    @_cuda_module.jit(device=True)
    def _gpu_multiply_gf2n_device(a, b, n, irreducible):
        """Умножение в GF(2^n) на GPU (device function)"""
        if a == 0 or b == 0:
            return 0
        
        result = 0
        # Умножение: для каждого бита b добавляем a << i
        for i in range(n):
            if b & (1 << i):
                result ^= a << i
        
        # Приведение по модулю неприводимого многочлена
        field_size = 1 << n
        while result >= field_size:
            # Находим степень старшего бита
            degree = 0
            t = result
            while t > 0:
                t >>= 1
                degree += 1
            # Сдвиг для приведения: degree - n - 1
            shift = degree - n - 1
            if shift >= 0:
                result ^= irreducible << shift
            else:
                break
        
        return result
    
    @_cuda_module.jit(device=True)
    def _gpu_power_gf2n_device(base, exp, n, irreducible):
        """Возведение в степень в GF(2^n) на GPU (device function)"""
        if exp == 0:
            return 1
        if exp == 1:
            return base
        
        result = 1
        while exp > 0:
            if exp & 1:
                result = _gpu_multiply_gf2n_device(result, base, n, irreducible)
            base = _gpu_multiply_gf2n_device(base, base, n, irreducible)
            exp >>= 1
        
        return result
else:
    # Заглушки для случая, когда CUDA недоступен
    def _gpu_multiply_gf2n_device(a, b, n, irreducible):
        return 0
    
    def _gpu_power_gf2n_device(base, exp, n, irreducible):
        return 1


class HFEGPUParallel(HFEBase):
    """
    GPU-параллельная реализация HFE с использованием CUDA
    """
    
    def __init__(self, n: int = 8, d: int = 3, seed: Optional[int] = None,
                 threads_per_block: Optional[int] = None):
        """
        Инициализация GPU-параллельного HFE
        
        Args:
            n: Размерность поля (GF(2^n))
            d: Степень многочлена HFE
            seed: Seed для генерации случайных ключей
            threads_per_block: Количество потоков на блок (для CUDA). 
                              Если None, выбирается автоматически (128 для лучшей занятости)
        """
        if not CUDA_AVAILABLE:
            error_msg = (
                "CUDA недоступен. GPU-параллелизация требует:\n"
                "1. NVIDIA GPU с поддержкой CUDA\n"
                "2. Установленный CUDA Toolkit\n"
                "3. Установленный Numba с поддержкой CUDA: pip install numba\n\n"
                "Примечание: На macOS с Apple Silicon или без NVIDIA GPU CUDA недоступен.\n"
                "Используйте CPU-параллельную версию вместо GPU."
            )
            raise RuntimeError(error_msg)
        
        super().__init__(n, d, seed)
        # Автоматический выбор оптимального threads_per_block для лучшей занятости
        if threads_per_block is None:
            # Используем 64 для лучшей занятости (меньше потоков = больше блоков)
            # Это позволяет создать больше блоков и лучше утилизировать GPU
            self.threads_per_block = 64
        else:
            self.threads_per_block = threads_per_block
        logger.info(f"Инициализация GPU-параллельного HFE с {self.threads_per_block} потоками на блок")
        
        # Преобразование матриц в массивы для GPU
        try:
            self._prepare_gpu_data()
        except Exception as e:
            error_msg = (
                f"Ошибка при инициализации GPU: {e}\n\n"
                "Возможные причины:\n"
                "1. CUDA драйвер не установлен или не найден\n"
                "2. NVIDIA GPU недоступен (например, на macOS)\n"
                "3. Неправильная версия CUDA Toolkit\n\n"
                "Решение: Используйте CPU-параллельную версию (--mode cpu)"
            )
            raise RuntimeError(error_msg) from e
    
    def _prepare_gpu_data(self):
        """Подготовка данных для передачи на GPU"""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA недоступен")
        
        logger.debug("Подготовка данных для GPU...")
        
        # Преобразование матриц в numpy массивы
        self.S1_gpu = cuda.to_device(self.S1.astype(np.int32))
        # S0 и T0 - это байты, но в аффинном преобразовании используются как биты (по модулю 2)
        self.S0_gpu = cuda.to_device((self.S0 % 2).astype(np.int32))
        self.T1_gpu = cuda.to_device(self.T1.astype(np.int32))
        self.T0_gpu = cuda.to_device((self.T0 % 2).astype(np.int32))
        self.S1_inv_gpu = cuda.to_device(self.S1_inv.astype(np.int32))
        self.T1_inv_gpu = cuda.to_device(self.T1_inv.astype(np.int32))
        
        logger.debug("Данные подготовлены для GPU")
    
    @staticmethod
    @cuda_jit
    def _gpu_bits_to_field(bits, n, n_bytes, field_elements, total_threads):
        """Преобразование бит в элементы поля на GPU"""
        idx = cuda.grid(1)
        # Обрабатываем несколько байтов на поток для лучшей утилизации (grid-stride loop)
        byte_idx = idx
        while byte_idx < n_bytes:
            val = 0
            for j in range(n):
                if bits[byte_idx * n + j] == 1:
                    val |= (1 << j)
            field_elements[byte_idx] = val
            byte_idx += total_threads
    
    @staticmethod
    @cuda_jit
    def _gpu_field_to_bits(field_elements, n, n_bytes, bits, total_threads):
        """Преобразование элементов поля в биты на GPU"""
        idx = cuda.grid(1)
        # Обрабатываем несколько байтов на поток для лучшей утилизации (grid-stride loop)
        byte_idx = idx
        while byte_idx < n_bytes:
            val = field_elements[byte_idx]
            for j in range(n):
                bits[byte_idx * n + j] = (val >> j) & 1
            byte_idx += total_threads
    
    @staticmethod
    @cuda_jit
    def _gpu_hfe_polynomial(x, n, d, result, total_threads):
        """Вычисление HFE многочлена на GPU: P(x) = sum(x^(2^i)) для i=1..d"""
        idx = cuda.grid(1)
        # Обрабатываем несколько элементов на поток для лучшей утилизации (grid-stride loop)
        elem_idx = idx
        while elem_idx < len(x):
            res = 0
            field_size = 1 << n
            irreducible = 0x11B if n == 8 else (1 << n) | 1
            val = x[elem_idx]
            
            for i in range(1, d + 1):
                power = 1 << i  # 2^i
                if power < field_size:
                    # Вычисление val^(2^i) через правильное возведение в степень
                    temp = _gpu_power_gf2n_device(val, power, n, irreducible)
                    res ^= temp
            result[elem_idx] = res
            elem_idx += total_threads
    
    @staticmethod
    @cuda_jit
    def _gpu_affine_transform(input_data, A, b, n, output, n_bytes, total_threads):
        """Аффинное преобразование на GPU: y = A*x + b для каждого байта"""
        idx = cuda.grid(1)
        # Обрабатываем несколько байтов на поток для лучшей утилизации (grid-stride loop)
        # Каждый поток обрабатывает байты: idx, idx + total_threads, idx + 2*total_threads, ...
        byte_idx = idx
        while byte_idx < n_bytes:
            # Для каждого байта (byte_idx) обрабатываем вектор из n бит
            for i in range(n):
                sum_val = 0
                for j in range(n):
                    # XOR эквивалентен сложению по модулю 2
                    if A[i, j] == 1:
                        sum_val ^= input_data[byte_idx * n + j]
                # Добавляем смещение b[i]
                output[byte_idx * n + i] = (sum_val ^ b[i]) % 2
            byte_idx += total_threads
    
    @staticmethod
    @cuda_jit
    def _gpu_solve_hfe(y, n, d, result):
        """Решение HFE уравнения на GPU (перебор)"""
        idx = cuda.grid(1)
        if idx < len(y):
            field_size = 1 << n
            target = y[idx]
            
            # Перебор всех возможных значений
            found = 0
            for x in range(field_size):
                # Вычисление HFE многочлена
                res = 0
                for i in range(1, d + 1):
                    power = 1 << i
                    if power < field_size:
                        val = x
                        for _ in range(i):
                            val = (val << 1) if val < (field_size >> 1) else ((val << 1) ^ 0x11B)
                        res ^= val
                
                if res == target:
                    found = x
                    break
            
            result[idx] = found
    
    def encrypt_block(self, data: bytes) -> bytes:
        """
        Параллельное шифрование блока данных на GPU
        
        Args:
            data: Байты для шифрования
        
        Returns:
            Зашифрованные байты
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA недоступен")
        
        logger.info(f"GPU-параллельное шифрование блока данных размером {len(data)} байт")
        
        data_array = np.array(list(data), dtype=np.uint8)
        n_bytes = len(data_array)
        
        # Преобразование байтов в векторы бит
        input_bits = np.zeros(n_bytes * self.n, dtype=np.int32)
        for i, byte_val in enumerate(data_array):
            for j in range(self.n):
                input_bits[i * self.n + j] = (byte_val >> j) & 1
        
        # Выделение памяти на GPU
        d_input = cuda.to_device(input_bits)
        d_intermediate1 = cuda.device_array(n_bytes * self.n, dtype=np.int32)
        d_intermediate2 = cuda.device_array(n_bytes, dtype=np.int32)
        d_intermediate3 = cuda.device_array(n_bytes * self.n, dtype=np.int32)
        d_output = cuda.device_array(n_bytes * self.n, dtype=np.int32)
        
        # Настройка grid и block для оптимальной занятости GPU
        # Используем минимум 128 блоков для лучшей утилизации GPU
        # Современные GPU требуют много блоков для хорошей занятости
        min_blocks = 128
        blocks_per_grid = max(min_blocks, (n_bytes + self.threads_per_block - 1) // self.threads_per_block)
        total_threads = blocks_per_grid * self.threads_per_block
        
        # Шаг 1: Применение S
        self._gpu_affine_transform[blocks_per_grid, self.threads_per_block](
            d_input, self.S1_gpu, self.S0_gpu, self.n, d_intermediate1, n_bytes, total_threads
        )
        cuda.synchronize()
        
        # Преобразование в элементы поля на GPU
        d_field_elements = cuda.device_array(n_bytes, dtype=np.int32)
        self._gpu_bits_to_field[blocks_per_grid, self.threads_per_block](
            d_intermediate1, self.n, n_bytes, d_field_elements, total_threads
        )
        cuda.synchronize()
        
        # Шаг 2: Применение HFE многочлена
        self._gpu_hfe_polynomial[blocks_per_grid, self.threads_per_block](
            d_field_elements, self.n, self.d, d_intermediate2, total_threads
        )
        cuda.synchronize()
        
        # Преобразование обратно в биты на GPU
        d_field_bits = cuda.device_array(n_bytes * self.n, dtype=np.int32)
        self._gpu_field_to_bits[blocks_per_grid, self.threads_per_block](
            d_intermediate2, self.n, n_bytes, d_field_bits, total_threads
        )
        cuda.synchronize()
        
        # Шаг 3: Применение T
        self._gpu_affine_transform[blocks_per_grid, self.threads_per_block](
            d_field_bits, self.T1_gpu, self.T0_gpu, self.n, d_output, n_bytes, total_threads
        )
        cuda.synchronize()
        
        # Копирование результатов обратно
        output_bits = d_output.copy_to_host()
        
        # Преобразование обратно в байты
        result_bytes = []
        for i in range(n_bytes):
            byte_val = 0
            for j in range(self.n):
                byte_val |= (output_bits[i * self.n + j] << j)
            result_bytes.append(byte_val)
        
        logger.info(f"GPU-параллельное шифрование завершено, получено {len(result_bytes)} байт")
        return bytes(result_bytes)
    
    def decrypt_block(self, data: bytes) -> bytes:
        """
        Параллельное расшифрование блока данных на GPU
        
        Args:
            data: Зашифрованные байты
        
        Returns:
            Расшифрованные байты
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA недоступен")
        
        logger.info(f"GPU-параллельное расшифрование блока данных размером {len(data)} байт")
        
        # Для упрощения используем CPU версию, так как решение HFE требует перебора
        # В реальной реализации можно оптимизировать
        logger.warning("GPU расшифрование использует гибридный подход (GPU + CPU)")
        
        data_array = np.array(list(data), dtype=np.uint8)
        n_bytes = len(data_array)
        
        # Преобразование байтов в векторы бит
        input_bits = np.zeros(n_bytes * self.n, dtype=np.int32)
        for i, byte_val in enumerate(data_array):
            for j in range(self.n):
                input_bits[i * self.n + j] = (byte_val >> j) & 1
        
        # Выделение памяти на GPU
        d_input = cuda.to_device(input_bits)
        d_intermediate1 = cuda.device_array(n_bytes * self.n, dtype=np.int32)
        d_output = cuda.device_array(n_bytes * self.n, dtype=np.int32)
        
        # Настройка grid и block для оптимальной занятости GPU
        # Используем минимум 128 блоков для лучшей утилизации GPU
        # Современные GPU требуют много блоков для хорошей занятости
        min_blocks = 128
        blocks_per_grid = max(min_blocks, (n_bytes + self.threads_per_block - 1) // self.threads_per_block)
        total_threads = blocks_per_grid * self.threads_per_block
        
        # Шаг 1: Обратное преобразование T: y = T1_inv * (x - T0) = T1_inv * x + T1_inv * (-T0)
        T0_neg = ((-self.T0) % 2).astype(np.int32)
        T1_inv_T0 = (self.T1_inv @ T0_neg) % 2
        T1_inv_T0_gpu = cuda.to_device(T1_inv_T0.astype(np.int32))
        
        self._gpu_affine_transform[blocks_per_grid, self.threads_per_block](
            d_input, self.T1_inv_gpu, T1_inv_T0_gpu, self.n, d_intermediate1, n_bytes, total_threads
        )
        cuda.synchronize()
        
        # Преобразование в элементы поля на GPU
        d_field_elements = cuda.device_array(n_bytes, dtype=np.int32)
        self._gpu_bits_to_field[blocks_per_grid, self.threads_per_block](
            d_intermediate1, self.n, n_bytes, d_field_elements, total_threads
        )
        cuda.synchronize()
        
        # Решение HFE на CPU (так как требует перебора)
        field_elements = d_field_elements.copy_to_host()
        solved = np.zeros(n_bytes, dtype=np.int32)
        for i in range(n_bytes):
            solved[i] = self._solve_hfe(int(field_elements[i]))
        
        # Преобразование обратно в биты на GPU
        d_solved = cuda.to_device(solved.astype(np.int32))
        d_solved_bits = cuda.device_array(n_bytes * self.n, dtype=np.int32)
        self._gpu_field_to_bits[blocks_per_grid, self.threads_per_block](
            d_solved, self.n, n_bytes, d_solved_bits, total_threads
        )
        cuda.synchronize()
        
        # Шаг 3: Обратное преобразование S: x = S1_inv * (y - S0) = S1_inv * y + S1_inv * (-S0)
        S0_neg = ((-self.S0) % 2).astype(np.int32)
        S1_inv_S0 = (self.S1_inv @ S0_neg) % 2
        S1_inv_S0_gpu = cuda.to_device(S1_inv_S0.astype(np.int32))
        
        self._gpu_affine_transform[blocks_per_grid, self.threads_per_block](
            d_solved_bits, self.S1_inv_gpu, S1_inv_S0_gpu, self.n, d_output, n_bytes, total_threads
        )
        cuda.synchronize()
        
        # Копирование результатов
        output_bits = d_output.copy_to_host()
        
        # Преобразование обратно в байты
        result_bytes = []
        for i in range(n_bytes):
            byte_val = 0
            for j in range(self.n):
                byte_val |= (output_bits[i * self.n + j] << j)
            result_bytes.append(byte_val)
        
        logger.info(f"GPU-параллельное расшифрование завершено, получено {len(result_bytes)} байт")
        return bytes(result_bytes)

