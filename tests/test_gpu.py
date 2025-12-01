#!/usr/bin/env python3
"""
Тест GPU-параллельной версии HFE
"""
import sys
import os
from pathlib import Path

# Добавление корневой директории проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import subprocess

try:
    from src.hfe_gpu_parallel import CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False


def main():
    """Запуск теста GPU-параллельной версии"""
    print("=" * 50)
    print("Тест GPU-параллельной версии HFE")
    print("=" * 50)
    
    # Проверка наличия main.py
    main_py = project_root / "main.py"
    if not main_py.exists():
        print("ОШИБКА: main.py не найден")
        return 1
    
    # Проверка доступности CUDA
    print("Проверка доступности CUDA...")
    if not CUDA_AVAILABLE:
        print("ПРЕДУПРЕЖДЕНИЕ: CUDA недоступен, тест будет пропущен")
        print("Для использования GPU-версии установите PyTorch с поддержкой CUDA")
        return 0
    
    print("Запуск GPU-параллельной версии с размером данных 65536 байт...")
    
    # Запуск теста
    result = subprocess.run(
        [
            sys.executable,
            str(main_py),
            "--mode", "gpu",
            "--data-size", "65536",
            "--log-level", "INFO"
        ],
        cwd=str(project_root)
    )
    
    if result.returncode == 0:
        print("\n✓ Тест GPU-параллельной версии пройден успешно")
        return 0
    else:
        print("\n✗ Тест GPU-параллельной версии провален")
        return 1


if __name__ == "__main__":
    sys.exit(main())

