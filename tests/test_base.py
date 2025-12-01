#!/usr/bin/env python3
"""
Тест базовой версии HFE (без распараллеливания)
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
    """Запуск теста базовой версии"""
    print("=" * 50)
    print("Тест базовой версии HFE")
    print("=" * 50)
    
    # Проверка наличия main.py
    main_py = project_root / "main.py"
    if not main_py.exists():
        print("ОШИБКА: main.py не найден")
        return 1
    
    print("Запуск базовой версии с размером данных 2048 байт...")
    
    # Запуск теста
    result = run_subprocess_safe(
        [
            sys.executable,
            str(main_py),
            "--mode", "base",
            "--data-size", "2048",
            "--log-level", "INFO"
        ],
        cwd=str(project_root)
    )
    
    if result.returncode == 0:
        print("\n✓ Тест базовой версии пройден успешно")
        return 0
    else:
        print("\n✗ Тест базовой версии провален")
        return 1


if __name__ == "__main__":
    sys.exit(main())

