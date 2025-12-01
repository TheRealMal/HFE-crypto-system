#!/usr/bin/env python3
"""
Тест CPU-параллельной версии HFE
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


def main():
    """Запуск теста CPU-параллельной версии"""
    print("=" * 50)
    print("Тест CPU-параллельной версии HFE")
    print("=" * 50)
    
    # Проверка наличия main.py
    main_py = project_root / "main.py"
    if not main_py.exists():
        print("ОШИБКА: main.py не найден")
        return 1
    
    print("Запуск CPU-параллельной версии с размером данных 2048 байт...")
    print("Использование 4 процессов...")
    
    # Запуск теста
    result = run_subprocess_safe(
        [
            sys.executable,
            str(main_py),
            "--mode", "cpu",
            "--data-size", "2048",
            "--num-processes", "4",
            "--log-level", "INFO"
        ],
        cwd=str(project_root)
    )
    
    if result.returncode == 0:
        print("\n✓ Тест CPU-параллельной версии пройден успешно")
        return 0
    else:
        print("\n✗ Тест CPU-параллельной версии провален")
        return 1


if __name__ == "__main__":
    sys.exit(main())

