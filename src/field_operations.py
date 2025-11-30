"""
Операции над конечными полями GF(2^n) для HFE
"""
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GF2n:
    """
    Класс для работы с конечным полем GF(2^n)
    """
    
    # Таблица известных неприводимых многочленов для малых n
    # Формат: n: (многочлен в виде целого числа, описание)
    KNOWN_IRREDUCIBLE = {
        1: (0x3, "x + 1"),
        2: (0x7, "x^2 + x + 1"),
        3: (0xB, "x^3 + x + 1"),
        4: (0x13, "x^4 + x + 1"),
        5: (0x25, "x^5 + x^2 + 1"),
        6: (0x43, "x^6 + x + 1"),
        7: (0x83, "x^7 + x + 1"),
        8: (0x11B, "x^8 + x^4 + x^3 + x + 1"),
        9: (0x211, "x^9 + x + 1"),
        10: (0x409, "x^10 + x^3 + 1"),
        11: (0x805, "x^11 + x^2 + 1"),
        12: (0x1053, "x^12 + x^3 + x^2 + x + 1"),
        13: (0x201B, "x^13 + x^4 + x^3 + x + 1"),
        14: (0x402B, "x^14 + x^5 + x^3 + x + 1"),
        15: (0x8003, "x^15 + x + 1"),
        16: (0x1002D, "x^16 + x^5 + x^3 + x^2 + 1"),
    }
    
    def __init__(self, n: int = 8):
        """
        Инициализация поля GF(2^n)
        
        Args:
            n: Размерность поля (по умолчанию 8, т.е. GF(2^8))
        """
        self.n = n
        self.field_size = 2 ** n
        logger.debug(f"Инициализация поля GF(2^{n}), размер: {self.field_size}")
        
        # Поиск неприводимого многочлена
        self.irreducible = self._find_irreducible_polynomial()
        logger.info(f"Используется неприводимый многочлен: {self.irreducible:0{self.n+1}b} (0x{self.irreducible:X})")
    
    def _find_irreducible_polynomial(self) -> int:
        """
        Поиск неприводимого многочлена степени n над GF(2)
        
        Returns:
            Неприводимый многочлен в виде целого числа
        """
        # Проверяем таблицу известных многочленов
        if self.n in self.KNOWN_IRREDUCIBLE:
            poly, desc = self.KNOWN_IRREDUCIBLE[self.n]
            logger.debug(f"Использован известный неприводимый многочлен: {desc}")
            return poly
        
        # Для больших n ищем алгоритмически
        logger.debug(f"Поиск неприводимого многочлена степени {self.n}...")
        return self._search_irreducible_polynomial()
    
    def _search_irreducible_polynomial(self) -> int:
        """
        Алгоритмический поиск неприводимого многочлена
        
        Returns:
            Неприводимый многочлен в виде целого числа
        """
        # Начинаем с простых кандидатов: x^n + x^k + 1 для малых k
        # Многочлен должен иметь вид: x^n + ... + 1 (старший и младший коэффициенты = 1)
        
        # Минимальный многочлен: x^n + 1
        min_poly = (1 << self.n) | 1
        
        # Пробуем многочлены вида x^n + x^k + 1 для k от 1 до n-1
        for k in range(1, min(self.n, 10)):  # Ограничиваем поиск для производительности
            candidate = (1 << self.n) | (1 << k) | 1
            if self._is_irreducible(candidate):
                logger.debug(f"Найден неприводимый многочлен: x^{self.n} + x^{k} + 1")
                return candidate
        
        # Если не нашли, пробуем более сложные формы
        for k1 in range(1, min(self.n, 8)):
            for k2 in range(k1 + 1, min(self.n, 8)):
                candidate = (1 << self.n) | (1 << k2) | (1 << k1) | 1
                if self._is_irreducible(candidate):
                    logger.debug(f"Найден неприводимый многочлен: x^{self.n} + x^{k2} + x^{k1} + 1")
                    return candidate
        
        # Если ничего не найдено, используем простой вариант (может быть приводимым, но работает)
        logger.warning(f"Не найден проверенный неприводимый многочлен для n={self.n}, используется x^{self.n} + x + 1")
        return (1 << self.n) | 2 | 1  # x^n + x + 1
    
    def _is_irreducible(self, poly: int) -> bool:
        """
        Проверка многочлена на неприводимость над GF(2)
        
        Многочлен неприводим, если он не делится ни на один многочлен степени от 1 до floor(n/2)
        
        Args:
            poly: Многочлен в виде целого числа
        
        Returns:
            True если многочлен неприводим, False иначе
        """
        if poly < 4:  # Многочлены степени 0 или 1 всегда неприводимы (или тривиальны)
            return True
        
        degree = poly.bit_length() - 1
        
        # Проверяем делимость на все многочлены степени от 1 до floor(degree/2)
        max_divisor_degree = degree // 2
        
        for d in range(1, max_divisor_degree + 1):
            # Генерируем все возможные делители степени d
            # Делитель должен иметь вид: x^d + ... + 1 (старший и младший коэффициенты = 1)
            min_divisor = (1 << d) | 1
            max_divisor = (1 << (d + 1)) - 1
            
            for divisor in range(min_divisor, max_divisor + 1):
                # Проверяем, что делитель действительно степени d
                if divisor.bit_length() - 1 != d:
                    continue
                
                # Проверяем делимость
                if self._polynomial_divide(poly, divisor)[1] == 0:
                    return False
        
        return True
    
    def _polynomial_divide(self, dividend: int, divisor: int) -> Tuple[int, int]:
        """
        Деление многочленов над GF(2)
        
        Args:
            dividend: Делимое (многочлен)
            divisor: Делитель (многочлен)
        
        Returns:
            Кортеж (частное, остаток)
        """
        if divisor == 0:
            raise ValueError("Деление на ноль")
        
        if dividend < divisor:
            return (0, dividend)
        
        quotient = 0
        remainder = dividend
        divisor_degree = divisor.bit_length() - 1
        
        while remainder >= divisor:
            remainder_degree = remainder.bit_length() - 1
            if remainder_degree < divisor_degree:
                break
            
            shift = remainder_degree - divisor_degree
            quotient ^= (1 << shift)
            remainder ^= (divisor << shift)
        
        return (quotient, remainder)
    
    def add(self, a: int, b: int) -> int:
        """Сложение в GF(2^n) - просто XOR"""
        return a ^ b
    
    def multiply(self, a: int, b: int) -> int:
        """
        Умножение в GF(2^n) с использованием неприводимого многочлена
        
        Args:
            a: Первый элемент поля
            b: Второй элемент поля
        
        Returns:
            Произведение a * b в поле GF(2^n)
        """
        if a == 0 or b == 0:
            return 0
        
        result = 0
        # Простое умножение с приведением по модулю
        for i in range(self.n):
            if b & (1 << i):
                result ^= a << i
        
        # Приведение по модулю неприводимого многочлена
        while result >= (1 << self.n):
            shift = result.bit_length() - self.n - 1
            result ^= self.irreducible << shift
        
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

