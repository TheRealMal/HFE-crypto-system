"""
Утилиты для тестов с поддержкой Windows
"""
import sys
import locale
import subprocess
from typing import List, Optional


def get_system_encoding():
    """
    Получение системной кодировки с учетом Windows
    
    Returns:
        str: Системная кодировка или 'utf-8' по умолчанию
    """
    try:
        # Попытка получить системную кодировку
        encoding = locale.getpreferredencoding()
        if encoding:
            return encoding
    except:
        pass
    
    # Fallback на UTF-8
    return 'utf-8'


def run_subprocess_safe(
    args: List[str],
    cwd: Optional[str] = None,
    capture_output: bool = False,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Безопасный запуск subprocess с правильной обработкой кодировки для Windows
    
    Args:
        args: Аргументы для subprocess.run
        cwd: Рабочая директория
        capture_output: Захватывать ли вывод
        **kwargs: Дополнительные аргументы для subprocess.run
    
    Returns:
        subprocess.CompletedProcess: Результат выполнения
    """
    # Определение кодировки
    encoding = get_system_encoding()
    
    # Настройка параметров для Windows
    subprocess_kwargs = {
        'cwd': cwd,
        **kwargs
    }
    
    # Проверяем, нужно ли захватывать вывод
    needs_encoding = (
        capture_output or 
        kwargs.get('stdout') is not None or 
        kwargs.get('stderr') is not None
    )
    
    if needs_encoding:
        # Если нужно захватить вывод, используем правильную кодировку
        subprocess_kwargs['encoding'] = encoding
        subprocess_kwargs['errors'] = 'replace'  # Заменяем нечитаемые символы
        if 'text' not in subprocess_kwargs:
            subprocess_kwargs['text'] = True
    
    return subprocess.run(args, **subprocess_kwargs)

