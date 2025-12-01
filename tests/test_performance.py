#!/usr/bin/env python3
"""
Тест производительности всех версий HFE
"""
import sys
import os
from pathlib import Path
from datetime import datetime

# Добавление корневой директории проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess


def main():
    """Запуск тестов производительности"""
    print("=" * 50)
    print("Тест производительности HFE")
    print("=" * 50)
    
    # Проверка наличия main.py
    main_py = project_root / "main.py"
    if not main_py.exists():
        print("ОШИБКА: main.py не найден")
        return 1
    
    # Размеры данных для тестирования
    sizes = [
        2**8, 2**9, 2**10,
        2**11, 2**12, 2**13,
        2**14, 2**15, 2**16,
        2**17
    ]
    
    print("Тестирование производительности с разными размерами данных...")
    print()
    
    # Создание директории для результатов
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    results_file = logs_dir / "performance_results.log"
    
    # Очистка файла результатов
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("Результаты тестирования производительности\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
    
    for size in sizes:
        print(f"Тестирование с размером данных: {size} байт")
        
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(f"Размер данных: {size} байт\n")
            f.write("-" * 50 + "\n")
        
        # Запуск теста
        result = subprocess.run(
            [
                sys.executable,
                str(main_py),
                "--mode", "all",
                "--data-size", str(size),
                "--log-level", "INFO"
            ],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
        with open(results_file, 'a', encoding='utf-8') as f:
            f.write(result.stdout)
            f.write("\n\n")
    
    print()
    print("=" * 50)
    print("Тестирование завершено")
    print(f"Результаты сохранены в: {results_file}")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

