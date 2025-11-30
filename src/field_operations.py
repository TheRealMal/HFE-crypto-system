"""
Операции над конечными полями GF(2^n) для HFE
"""
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class GF2n:
    """
    Класс для работы с конечным полем GF(2^n)
    """
    
    def __init__(self, n: int = 8):
        """
        Инициализация поля GF(2^n)
        
        Args:
            n: Размерность поля (по умолчанию 8, т.е. GF(2^8))
        """
        self.n = n
        self.field_size = 2 ** n
        logger.debug(f"Инициализация поля GF(2^{n}), размер: {self.field_size}")
    
    def add(self, a: int, b: int) -> int:
        """Сложение в GF(2^n) - просто XOR"""
        return a ^ b
    
    def multiply(self, a: int, b: int) -> int:
        """
        Умножение в GF(2^n) с использованием неприводимого многочлена
        Для GF(2^8) используется многочлен x^8 + x^4 + x^3 + x + 1
        """
        if a == 0 or b == 0:
            return 0
        
        result = 0
        # Простое умножение с приведением по модулю
        for i in range(self.n):
            if b & (1 << i):
                result ^= a << i
        
        # Приведение по модулю неприводимого многочлена
        if self.n == 8:
            # Многочлен: x^8 + x^4 + x^3 + x + 1 = 0x11B
            irreducible = 0x11B
            while result >= (1 << self.n):
                shift = result.bit_length() - self.n - 1
                result ^= irreducible << shift
        
        return result
    
    def power(self, a: int, exp: int) -> int:
        """Возведение в степень в GF(2^n)"""
        if exp == 0:
            return 1
        if exp == 1:
            return a
        
        result = 1
        while exp > 0:
            if exp & 1:
                result = self.multiply(result, a)
            a = self.multiply(a, a)
            exp >>= 1
        
        return result
    
    def vector_to_field(self, vector: List[int]) -> int:
        """Преобразование вектора в элемент поля"""
        result = 0
        for i, bit in enumerate(vector):
            if bit:
                result |= (1 << i)
        return result
    
    def field_to_vector(self, element: int) -> List[int]:
        """Преобразование элемента поля в вектор"""
        return [(element >> i) & 1 for i in range(self.n)]

