#!/usr/bin/env python3
"""
Запуск всех тестов HFE
"""
import sys
import os
from pathlib import Path

# Добавление корневой директории проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Добавление директории tests в путь для импорта
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

from test_utils import run_subprocess_safe

# Цвета для вывода (ANSI escape codes)
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color


def run_test(test_name, test_script):
    """Запуск теста"""
    print("-" * 50)
    print(f"Запуск: {test_name}")
    print("-" * 50)
    
    test_path = project_root / test_script
    
    if not test_path.exists():
        print(f"{Colors.YELLOW}⚠ Тест не найден: {test_script}{Colors.NC}")
        return None  # Пропущен
    
    result = run_subprocess_safe(
        [sys.executable, str(test_path)],
        cwd=str(project_root)
    )
    
    if result.returncode == 0:
        print(f"{Colors.GREEN}✓ {test_name} пройден{Colors.NC}")
        return True
    else:
        print(f"{Colors.RED}✗ {test_name} провален{Colors.NC}")
        return False


def main():
    """Главная функция"""
    print("=" * 50)
    print("Запуск всех тестов HFE")
    print("=" * 50)
    print()
    
    # Счетчики
    passed = 0
    failed = 0
    skipped = 0
    
    # Список тестов
    tests = [
        ("Тест корректности", "tests/test_correctness.py"),
        ("Тест базовой версии", "tests/test_base.py"),
        ("Тест CPU-параллельной версии", "tests/test_cpu.py"),
        ("Тест GPU-параллельной версии", "tests/test_gpu.py"),
        ("Комплексный тест всех версий", "tests/test_all.py"),
    ]
    
    # Запуск тестов
    for test_name, test_script in tests:
        result = run_test(test_name, test_script)
        if result is None:
            skipped += 1
        elif result:
            passed += 1
        else:
            failed += 1
        print()
    
    # Итоги
    print("=" * 50)
    print("Итоги тестирования")
    print("=" * 50)
    print(f"{Colors.GREEN}Пройдено: {passed}{Colors.NC}")
    print(f"{Colors.RED}Провалено: {failed}{Colors.NC}")
    print(f"{Colors.YELLOW}Пропущено: {skipped}{Colors.NC}")
    print()
    
    if failed == 0:
        print(f"{Colors.GREEN}✓ Все тесты пройдены успешно!{Colors.NC}")
        return 0
    else:
        print(f"{Colors.RED}✗ Некоторые тесты провалены{Colors.NC}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

