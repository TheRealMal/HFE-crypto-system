"""
Базовая реализация HFE (Hidden Field Equations) без распараллеливания
"""
import numpy as np
from typing import List, Tuple, Optional
import logging
from .field_operations import GF2n

logger = logging.getLogger(__name__)


class HFEBase:
    """
    Базовая реализация криптосистемы HFE
    """
    
    def __init__(self, n: int = 8, d: int = 3, seed: Optional[int] = None):
        """
        Инициализация HFE
        
        Args:
            n: Размерность поля (GF(2^n))
            d: Степень многочлена HFE
            seed: Seed для генерации случайных ключей
        """
        self.n = n
        self.d = d
        self.field = GF2n(n)
        self.field_size = 2 ** n
        
        if seed is not None:
            np.random.seed(seed)
        
        logger.info(f"Инициализация HFE: n={n}, d={d}, размер поля={self.field_size}")
        
        # Генерация секретных ключей
        self._generate_keys()
    
    def _generate_keys(self):
        """Генерация секретных ключей: аффинные преобразования S и T"""
        logger.debug("Генерация секретных ключей...")
        
        # Аффинное преобразование S: y = S1*x + S0
        self.S1 = self._generate_invertible_matrix()
        self.S0 = np.random.randint(0, self.field_size, size=self.n, dtype=np.uint8)
        
        # Аффинное преобразование T: y = T1*x + T0
        self.T1 = self._generate_invertible_matrix()
        self.T0 = np.random.randint(0, self.field_size, size=self.n, dtype=np.uint8)
        
        # Обратные матрицы для расшифрования
        self.S1_inv = self._matrix_inverse(self.S1)
        self.T1_inv = self._matrix_inverse(self.T1)
        
        logger.debug("Ключи успешно сгенерированы")
    
    def _generate_invertible_matrix(self) -> np.ndarray:
        """Генерация обратимой матрицы над GF(2)"""
        while True:
            matrix = np.random.randint(0, 2, size=(self.n, self.n), dtype=np.uint8)
            # Проверка обратимости через определитель
            if self._is_invertible(matrix):
                return matrix
            logger.debug("Матрица не обратима, генерация новой...")
    
    def _is_invertible(self, matrix: np.ndarray) -> bool:
        """Проверка обратимости матрицы над GF(2)"""
        # Упрощенная проверка: матрица обратима если её ранг = n
        # Используем гауссово исключение
        matrix_copy = matrix.copy().astype(np.int32)
        rank = 0
        
        for col in range(self.n):
            # Поиск ненулевого элемента
            pivot_row = None
            for row in range(rank, self.n):
                if matrix_copy[row, col] % 2 == 1:
                    pivot_row = row
                    break
            
            if pivot_row is not None:
                # Перестановка строк
                if pivot_row != rank:
                    matrix_copy[[rank, pivot_row]] = matrix_copy[[pivot_row, rank]]
                
                # Исключение
                for row in range(rank + 1, self.n):
                    if matrix_copy[row, col] % 2 == 1:
                        matrix_copy[row] = (matrix_copy[row] + matrix_copy[rank]) % 2
                
                rank += 1
        
        return rank == self.n
    
    def _matrix_inverse(self, matrix: np.ndarray) -> np.ndarray:
        """Вычисление обратной матрицы над GF(2) методом Гаусса-Жордана"""
        n = matrix.shape[0]
        # Создаем расширенную матрицу [A|I]
        augmented = np.hstack([matrix.astype(np.int32), np.eye(n, dtype=np.int32)])
        
        # Прямой ход метода Гаусса
        for col in range(n):
            # Поиск опорного элемента
            pivot_row = None
            for row in range(col, n):
                if augmented[row, col] % 2 == 1:
                    pivot_row = row
                    break
            
            if pivot_row is None:
                raise ValueError("Матрица не обратима")
            
            # Перестановка строк
            if pivot_row != col:
                augmented[[col, pivot_row]] = augmented[[pivot_row, col]]
            
            # Исключение
            for row in range(n):
                if row != col and augmented[row, col] % 2 == 1:
                    augmented[row] = (augmented[row] + augmented[col]) % 2
        
        # Извлечение обратной матрицы
        inverse = augmented[:, n:].astype(np.uint8) % 2
        return inverse
    
    def _hfe_polynomial(self, x: int) -> int:
        """
        Вычисление значения HFE многочлена
        P(x) = sum(a_i * x^(2^i + 2^j)) для i <= j <= d
        """
        result = 0
        # Упрощенный HFE многочлен: x^2 + x^4 + ... + x^(2^d)
        for i in range(1, self.d + 1):
            power = 2 ** i
            if power < self.field_size:
                result = self.field.add(result, self.field.power(x, power))
        
        return result
    
    def _affine_transform(self, x: List[int], A: np.ndarray, b: np.ndarray) -> List[int]:
        """Применение аффинного преобразования: y = A*x + b"""
        x_vec = np.array(x, dtype=np.uint8)
        y = (A @ x_vec + b) % 2
        return y.tolist()
    
    def encrypt(self, plaintext: List[int]) -> List[int]:
        """
        Шифрование открытого текста
        
        Args:
            plaintext: Вектор из n бит
        
        Returns:
            Зашифрованный вектор из n бит
        """
        logger.debug(f"Шифрование: plaintext={plaintext}")
        
        # Шаг 1: Применение S
        x = self._affine_transform(plaintext, self.S1, self.S0)
        x_field = self.field.vector_to_field(x)
        
        # Шаг 2: Применение HFE многочлена
        y_field = self._hfe_polynomial(x_field)
        y = self.field.field_to_vector(y_field)
        
        # Шаг 3: Применение T
        ciphertext = self._affine_transform(y, self.T1, self.T0)
        
        logger.debug(f"Шифрование завершено: ciphertext={ciphertext}")
        return ciphertext
    
    def decrypt(self, ciphertext: List[int]) -> List[int]:
        """
        Расшифрование зашифрованного текста
        
        Args:
            ciphertext: Зашифрованный вектор из n бит
        
        Returns:
            Расшифрованный вектор из n бит
        """
        logger.debug(f"Расшифрование: ciphertext={ciphertext}")
        
        # Шаг 1: Обратное преобразование T
        y = self._affine_transform(ciphertext, self.T1_inv, 
                                   (self.T1_inv @ (-self.T0 % 2)) % 2)
        y_field = self.field.vector_to_field(y)
        
        # Шаг 2: Решение уравнения HFE (упрощенный вариант - перебор)
        x_field = self._solve_hfe(y_field)
        x = self.field.field_to_vector(x_field)
        
        # Шаг 3: Обратное преобразование S
        plaintext = self._affine_transform(x, self.S1_inv,
                                          (self.S1_inv @ (-self.S0 % 2)) % 2)
        
        logger.debug(f"Расшифрование завершено: plaintext={plaintext}")
        return plaintext
    
    def _solve_hfe(self, y: int) -> int:
        """
        Решение уравнения HFE: найти x такой, что P(x) = y
        Упрощенный вариант - перебор всех возможных значений
        """
        logger.debug(f"Решение HFE уравнения для y={y}")
        
        # Перебор всех возможных значений x
        for x in range(self.field_size):
            if self._hfe_polynomial(x) == y:
                logger.debug(f"Найдено решение: x={x}")
                return x
        
        # Если решение не найдено, возвращаем 0
        logger.warning(f"Решение не найдено для y={y}, возвращаем 0")
        return 0
    
    def encrypt_block(self, data: bytes) -> bytes:
        """
        Шифрование блока данных
        
        Args:
            data: Байты для шифрования
        
        Returns:
            Зашифрованные байты
        """
        logger.info(f"Шифрование блока данных размером {len(data)} байт")
        
        result = []
        for byte_val in data:
            # Преобразование байта в вектор бит
            plaintext = [(byte_val >> i) & 1 for i in range(self.n)]
            # Шифрование
            ciphertext = self.encrypt(plaintext)
            # Преобразование обратно в байт
            cipher_byte = sum(bit << i for i, bit in enumerate(ciphertext))
            result.append(cipher_byte)
        
        logger.info(f"Шифрование завершено, получено {len(result)} байт")
        return bytes(result)
    
    def decrypt_block(self, data: bytes) -> bytes:
        """
        Расшифрование блока данных
        
        Args:
            data: Зашифрованные байты
        
        Returns:
            Расшифрованные байты
        """
        logger.info(f"Расшифрование блока данных размером {len(data)} байт")
        
        result = []
        for byte_val in data:
            # Преобразование байта в вектор бит
            ciphertext = [(byte_val >> i) & 1 for i in range(self.n)]
            # Расшифрование
            plaintext = self.decrypt(ciphertext)
            # Преобразование обратно в байт
            plain_byte = sum(bit << i for i, bit in enumerate(plaintext))
            result.append(plain_byte)
        
        logger.info(f"Расшифрование завершено, получено {len(result)} байт")
        return bytes(result)

