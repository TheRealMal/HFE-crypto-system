#!/usr/bin/env python3
"""
Тест корректности работы HFE (проверка шифрование/расшифрование)
"""
import sys
import os
from pathlib import Path

# Добавление корневой директории проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hfe_base import HFEBase
from src.hfe_cpu_parallel import HFECPUParallel

try:
    from src.hfe_gpu_parallel import HFEGPUParallel, CUDA_AVAILABLE
except ImportError:
    CUDA_AVAILABLE = False
    HFEGPUParallel = None


def test_hfe_instance(hfe_instance, instance_name):
    """Тестирование экземпляра HFE"""
    # Тестовые данные
    test_cases = [
        b"Hello, World!",
        b"Test data 123",
        b"",
        bytes(range(256)),  # Все возможные байты
        b"A" * 100,  # Повторяющиеся символы
    ]
    
    print(f"Тестирование {instance_name}...")
    all_passed = True
    
    for i, data in enumerate(test_cases):
        try:
            encrypted = hfe_instance.encrypt_block(data)
            decrypted = hfe_instance.decrypt_block(encrypted)
            
            if data == decrypted:
                print(f"  ✓ Тест {i+1} пройден (размер: {len(data)} байт)")
            else:
                print(f"  ✗ Тест {i+1} провален (размер: {len(data)} байт)")
                print(f"     Ожидалось: {data[:20]}...")
                print(f"     Получено:  {decrypted[:20]}...")
                all_passed = False
        except Exception as e:
            print(f"  ✗ Тест {i+1} вызвал ошибку: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def main():
    """Запуск тестов корректности"""
    print("=" * 50)
    print("Тест корректности работы HFE")
    print("=" * 50)
    
    all_passed = True
    
    # Тест базовой версии
    print("\n" + "-" * 50)
    hfe_base = HFEBase(n=8, d=3, seed=42)
    if not test_hfe_instance(hfe_base, "базовой версии HFE"):
        all_passed = False
    
    # Тест CPU-параллельной версии
    print("\n" + "-" * 50)
    hfe_cpu = HFECPUParallel(n=8, d=3, seed=42, num_processes=2)
    if not test_hfe_instance(hfe_cpu, "CPU-параллельной версии HFE"):
        all_passed = False
    
    # Тест GPU-параллельной версии (если доступна)
    if CUDA_AVAILABLE and HFEGPUParallel is not None:
        print("\n" + "-" * 50)
        try:
            hfe_gpu = HFEGPUParallel(n=8, d=3, seed=42)
            if not test_hfe_instance(hfe_gpu, "GPU-параллельной версии HFE"):
                all_passed = False
        except Exception as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось создать GPU-версию: {e}")
            print("GPU тест пропущен")
    else:
        print("\n" + "-" * 50)
        print("GPU-версия недоступна, тест пропущен")
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ Все тесты корректности пройдены успешно")
        return 0
    else:
        print("✗ Некоторые тесты корректности провалены")
        return 1


if __name__ == "__main__":
    sys.exit(main())

