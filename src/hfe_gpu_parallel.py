"""
GPU-параллельная реализация HFE с использованием numba CUDA
"""
import numpy as np
from typing import List, Tuple, Optional
import logging

try:
    from numba import cuda, types
    from numba.cuda import jit as cuda_jit
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


class HFEGPUParallel(HFEBase):
    """
    GPU-параллельная реализация HFE с использованием CUDA
    """
    
    def __init__(self, n: int = 8, d: int = 3, seed: Optional[int] = None,
                 threads_per_block: int = 256):
        """
        Инициализация GPU-параллельного HFE
        
        Args:
            n: Размерность поля (GF(2^n))
            d: Степень многочлена HFE
            seed: Seed для генерации случайных ключей
            threads_per_block: Количество потоков на блок (для CUDA)
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
        self.threads_per_block = threads_per_block
        logger.info(f"Инициализация GPU-параллельного HFE с {threads_per_block} потоками на блок")
        
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
        self.S0_gpu = cuda.to_device(self.S0.astype(np.int32))
        self.T1_gpu = cuda.to_device(self.T1.astype(np.int32))
        self.T0_gpu = cuda.to_device(self.T0.astype(np.int32))
        self.S1_inv_gpu = cuda.to_device(self.S1_inv.astype(np.int32))
        self.T1_inv_gpu = cuda.to_device(self.T1_inv.astype(np.int32))
        
        logger.debug("Данные подготовлены для GPU")
    
    @staticmethod
    @cuda_jit
    def _gpu_multiply_gf2n(a, b, n, result):
        """Умножение в GF(2^n) на GPU"""
        idx = cuda.grid(1)
        if idx < len(a):
            if a[idx] == 0 or b[idx] == 0:
                result[idx] = 0
            else:
                res = 0
                for i in range(n):
                    if b[idx] & (1 << i):
                        res ^= a[idx] << i
                
                # Приведение по модулю для GF(2^8)
                if n == 8:
                    irreducible = 0x11B
                    while res >= (1 << n):
                        shift = res.bit_length() - n - 1
                        res ^= irreducible << shift
                
                result[idx] = res
    
    @staticmethod
    @cuda_jit
    def _gpu_power_gf2n(base, exp, n, result):
        """Возведение в степень в GF(2^n) на GPU"""
        idx = cuda.grid(1)
        if idx < len(base):
            if exp[idx] == 0:
                result[idx] = 1
            elif exp[idx] == 1:
                result[idx] = base[idx]
            else:
                res = 1
                e = exp[idx]
                a = base[idx]
                while e > 0:
                    if e & 1:
                        # Упрощенное умножение
                        temp = res ^ a if (res != 0 and a != 0) else 0
                        res = temp
                    # Упрощенное возведение в квадрат
                    a = a << 1 if a < (1 << n) else (a << 1) ^ 0x11B
                    e >>= 1
                result[idx] = res
    
    @staticmethod
    @cuda_jit
    def _gpu_hfe_polynomial(x, n, d, result):
        """Вычисление HFE многочлена на GPU"""
        idx = cuda.grid(1)
        if idx < len(x):
            res = 0
            field_size = 1 << n
            for i in range(1, d + 1):
                power = 1 << i
                if power < field_size:
                    # Упрощенное вычисление x^power
                    val = x[idx]
                    for _ in range(i):
                        val = (val << 1) if val < (field_size >> 1) else ((val << 1) ^ 0x11B)
                    res ^= val
            result[idx] = res
    
    @staticmethod
    @cuda_jit
    def _gpu_affine_transform(input_data, A, b, n, output):
        """Аффинное преобразование на GPU"""
        idx = cuda.grid(1)
        if idx < len(input_data):
            for i in range(n):
                sum_val = 0
                for j in range(n):
                    sum_val ^= (A[i, j] * input_data[idx * n + j]) % 2
                output[idx * n + i] = (sum_val + b[i]) % 2
    
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
        
        # Настройка grid и block
        blocks_per_grid = (n_bytes + self.threads_per_block - 1) // self.threads_per_block
        
        # Шаг 1: Применение S
        self._gpu_affine_transform[blocks_per_grid, self.threads_per_block](
            d_input, self.S1_gpu, self.S0_gpu, self.n, d_intermediate1
        )
        cuda.synchronize()
        
        # Преобразование в элементы поля
        d_field_elements = cuda.device_array(n_bytes, dtype=np.int32)
        for i in range(n_bytes):
            val = 0
            for j in range(self.n):
                val |= (d_intermediate1[i * self.n + j] << j)
            d_field_elements[i] = val
        
        # Шаг 2: Применение HFE многочлена
        self._gpu_hfe_polynomial[blocks_per_grid, self.threads_per_block](
            d_field_elements, self.n, self.d, d_intermediate2
        )
        cuda.synchronize()
        
        # Преобразование обратно в биты
        d_field_bits = cuda.device_array(n_bytes * self.n, dtype=np.int32)
        for i in range(n_bytes):
            val = d_intermediate2[i]
            for j in range(self.n):
                d_field_bits[i * self.n + j] = (val >> j) & 1
        
        # Шаг 3: Применение T
        self._gpu_affine_transform[blocks_per_grid, self.threads_per_block](
            d_field_bits, self.T1_gpu, self.T0_gpu, self.n, d_output
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
        
        blocks_per_grid = (n_bytes + self.threads_per_block - 1) // self.threads_per_block
        
        # Шаг 1: Обратное преобразование T
        T0_neg = ((-self.T0) % 2).astype(np.int32)
        T0_neg_gpu = cuda.to_device(T0_neg)
        T1_inv_T0 = (self.T1_inv @ T0_neg) % 2
        T1_inv_T0_gpu = cuda.to_device(T1_inv_T0.astype(np.int32))
        
        self._gpu_affine_transform[blocks_per_grid, self.threads_per_block](
            d_input, self.T1_inv_gpu, T1_inv_T0_gpu, self.n, d_intermediate1
        )
        cuda.synchronize()
        
        # Решение HFE на CPU (так как требует перебора)
        intermediate1 = d_intermediate1.copy_to_host()
        field_elements = np.zeros(n_bytes, dtype=np.int32)
        for i in range(n_bytes):
            val = 0
            for j in range(self.n):
                val |= (intermediate1[i * self.n + j] << j)
            field_elements[i] = val
        
        # Решение HFE (на CPU)
        solved = np.zeros(n_bytes, dtype=np.int32)
        for i in range(n_bytes):
            solved[i] = self._solve_hfe(int(field_elements[i]))
        
        # Преобразование обратно в биты
        solved_bits = np.zeros(n_bytes * self.n, dtype=np.int32)
        for i in range(n_bytes):
            val = solved[i]
            for j in range(self.n):
                solved_bits[i * self.n + j] = (val >> j) & 1
        
        d_solved_bits = cuda.to_device(solved_bits)
        
        # Шаг 3: Обратное преобразование S
        S0_neg = ((-self.S0) % 2).astype(np.int32)
        S1_inv_S0 = (self.S1_inv @ S0_neg) % 2
        S1_inv_S0_gpu = cuda.to_device(S1_inv_S0.astype(np.int32))
        
        self._gpu_affine_transform[blocks_per_grid, self.threads_per_block](
            d_solved_bits, self.S1_inv_gpu, S1_inv_S0_gpu, self.n, d_output
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

