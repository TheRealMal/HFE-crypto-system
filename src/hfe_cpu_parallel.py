"""
CPU-параллельная реализация HFE с использованием multiprocessing
"""
import numpy as np
from typing import List, Tuple, Optional
import logging
from multiprocessing import Pool, cpu_count
from .hfe_base import HFEBase
from .field_operations import GF2n

logger = logging.getLogger(__name__)


def _encrypt_chunk_worker(args):
    """Рабочая функция для шифрования chunk (для multiprocessing)"""
    chunk, S1, S0, T1, T0, n, d, field_size = args
    
    field = GF2n(n)
    results = []
    
    for plaintext in chunk:
        # Шаг 1: Применение S
        x = _affine_transform(plaintext, S1, S0, n)
        x_field = field.vector_to_field(x)
        
        # Шаг 2: Применение HFE многочлена
        y_field = _hfe_polynomial(x_field, n, d, field_size)
        y = field.field_to_vector(y_field)
        
        # Шаг 3: Применение T
        ciphertext = _affine_transform(y, T1, T0, n)
        results.append(ciphertext)
    
    return results


def _decrypt_chunk_worker(args):
    """Рабочая функция для расшифрования chunk (для multiprocessing)"""
    chunk, S1_inv, S0, T1_inv, T0, n, d, field_size = args
    
    field = GF2n(n)
    results = []
    
    T0_neg = (-T0) % 2
    T1_inv_T0 = (T1_inv @ T0_neg) % 2
    S0_neg = (-S0) % 2
    S1_inv_S0 = (S1_inv @ S0_neg) % 2
    
    for ciphertext in chunk:
        # Шаг 1: Обратное преобразование T
        y = _affine_transform(ciphertext, T1_inv, T1_inv_T0, n)
        y_field = field.vector_to_field(y)
        
        # Шаг 2: Решение уравнения HFE
        x_field = _solve_hfe(y_field, n, d, field_size)
        x = field.field_to_vector(x_field)
        
        # Шаг 3: Обратное преобразование S
        plaintext = _affine_transform(x, S1_inv, S1_inv_S0, n)
        results.append(plaintext)
    
    return results


def _affine_transform(x: List[int], A: np.ndarray, b: np.ndarray, n: int) -> List[int]:
    """Применение аффинного преобразования: y = A*x + b"""
    x_vec = np.array(x, dtype=np.uint8)
    y = (A @ x_vec + b) % 2
    return y.tolist()


def _hfe_polynomial(x: int, n: int, d: int, field_size: int) -> int:
    """Вычисление значения HFE многочлена"""
    field = GF2n(n)
    result = 0
    for i in range(1, d + 1):
        power = 2 ** i
        if power < field_size:
            result = field.add(result, field.power(x, power))
    return result


def _solve_hfe(y: int, n: int, d: int, field_size: int) -> int:
    """Решение уравнения HFE: найти x такой, что P(x) = y"""
    for x in range(field_size):
        if _hfe_polynomial(x, n, d, field_size) == y:
            return x
    return 0


class HFECPUParallel(HFEBase):
    """
    CPU-параллельная реализация HFE с использованием multiprocessing
    """
    
    def __init__(self, n: int = 8, d: int = 3, seed: Optional[int] = None, 
                 num_processes: Optional[int] = None):
        """
        Инициализация CPU-параллельного HFE
        
        Args:
            n: Размерность поля (GF(2^n))
            d: Степень многочлена HFE
            seed: Seed для генерации случайных ключей
            num_processes: Количество процессов (по умолчанию - количество ядер CPU)
        """
        super().__init__(n, d, seed)
        self.num_processes = num_processes or cpu_count()
        logger.info(f"Инициализация CPU-параллельного HFE с {self.num_processes} процессами")
    
    def encrypt_block(self, data: bytes) -> bytes:
        """
        Параллельное шифрование блока данных на CPU
        
        Args:
            data: Байты для шифрования
        
        Returns:
            Зашифрованные байты
        """
        logger.info(f"CPU-параллельное шифрование блока данных размером {len(data)} байт")
        
        # Преобразование байтов в векторы бит
        plaintexts = [[(byte_val >> i) & 1 for i in range(self.n)] 
                      for byte_val in data]
        
        # Разделение на chunks для параллельной обработки
        chunk_size = max(1, len(plaintexts) // self.num_processes)
        chunks = [plaintexts[i:i + chunk_size] 
                  for i in range(0, len(plaintexts), chunk_size)]
        
        logger.debug(f"Разделено на {len(chunks)} chunks для обработки")
        
        # Параллельная обработка
        with Pool(processes=self.num_processes) as pool:
            # Подготовка аргументов для каждого chunk
            encrypt_args = [
                (chunk, self.S1, self.S0, self.T1, self.T0, 
                 self.n, self.d, self.field_size)
                for chunk in chunks
            ]
            results = pool.map(_encrypt_chunk_worker, encrypt_args)
        
        # Объединение результатов
        ciphertexts = []
        for chunk_result in results:
            ciphertexts.extend(chunk_result)
        
        # Преобразование обратно в байты
        result_bytes = []
        for ciphertext in ciphertexts:
            cipher_byte = sum(bit << i for i, bit in enumerate(ciphertext))
            result_bytes.append(cipher_byte)
        
        logger.info(f"CPU-параллельное шифрование завершено, получено {len(result_bytes)} байт")
        return bytes(result_bytes)
    
    def decrypt_block(self, data: bytes) -> bytes:
        """
        Параллельное расшифрование блока данных на CPU
        
        Args:
            data: Зашифрованные байты
        
        Returns:
            Расшифрованные байты
        """
        logger.info(f"CPU-параллельное расшифрование блока данных размером {len(data)} байт")
        
        # Преобразование байтов в векторы бит
        ciphertexts = [[(byte_val >> i) & 1 for i in range(self.n)] 
                       for byte_val in data]
        
        # Разделение на chunks для параллельной обработки
        chunk_size = max(1, len(ciphertexts) // self.num_processes)
        chunks = [ciphertexts[i:i + chunk_size] 
                    for i in range(0, len(ciphertexts), chunk_size)]
        
        logger.debug(f"Разделено на {len(chunks)} chunks для обработки")
        
        # Параллельная обработка
        with Pool(processes=self.num_processes) as pool:
            # Подготовка аргументов для каждого chunk
            decrypt_args = [
                (chunk, self.S1_inv, self.S0, self.T1_inv, self.T0,
                 self.n, self.d, self.field_size)
                for chunk in chunks
            ]
            results = pool.map(_decrypt_chunk_worker, decrypt_args)
        
        # Объединение результатов
        plaintexts = []
        for chunk_result in results:
            plaintexts.extend(chunk_result)
        
        # Преобразование обратно в байты
        result_bytes = []
        for plaintext in plaintexts:
            plain_byte = sum(bit << i for i, bit in enumerate(plaintext))
            result_bytes.append(plain_byte)
        
        logger.info(f"CPU-параллельное расшифрование завершено, получено {len(result_bytes)} байт")
        return bytes(result_bytes)

