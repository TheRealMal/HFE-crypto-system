#!/usr/bin/env python3
"""
Комплексный тест всех версий HFE
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
    """Запуск комплексного теста всех версий"""
    print("=" * 50)
    print("Комплексный тест всех версий HFE")
    print("=" * 50)
    
    # Проверка наличия main.py
    main_py = project_root / "main.py"
    if not main_py.exists():
        print("ОШИБКА: main.py не найден")
        return 1
    
    print("Запуск всех версий с размером данных 16000 байт...")
    print()
    
    # Создание директории для логов
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    log_file = logs_dir / "test_all.log"
    
    # Запуск всех режимов
    result = run_subprocess_safe(
        [
            sys.executable,
            str(main_py),
            "--mode", "all",
            "--data-size", "16000",
            "--log-level", "INFO",
            "--log-file", str(log_file)
        ],
        cwd=str(project_root)
    )
    
    if result.returncode == 0:
        print()
        print("=" * 50)
        print("✓ Все тесты пройдены успешно")
        print("=" * 50)
        return 0
    else:
        print()
        print("=" * 50)
        print("✗ Некоторые тесты провалены")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())

