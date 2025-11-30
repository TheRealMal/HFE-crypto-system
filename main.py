#!/usr/bin/env python3
"""
Главный файл для демонстрации работы HFE криптосистемы
Поддерживает три режима: обычный, CPU-параллельный, GPU-параллельный
"""
import argparse
import time
import sys
from pathlib import Path

# Добавление пути к модулям
sys.path.insert(0, str(Path(__file__).parent))

from src.logger_config import setup_logger
from src.hfe_base import HFEBase
from src.hfe_cpu_parallel import HFECPUParallel
from src.hfe_gpu_parallel import HFEGPUParallel, CUDA_AVAILABLE

logger = setup_logger("HFE_Main", log_level="INFO")


def benchmark_encryption_decryption(hfe_instance, data: bytes, mode: str):
    """
    Бенчмарк шифрования и расшифрования
    
    Args:
        hfe_instance: Экземпляр HFE
        data: Данные для обработки
        mode: Режим работы (для логирования)
    
    Returns:
        Кортеж (время шифрования, время расшифрования, успешность)
    """
    logger.info(f"=== Бенчмарк для режима: {mode} ===")
    logger.info(f"Размер данных: {len(data)} байт")
    
    # Шифрование
    start_time = time.time()
    encrypted = hfe_instance.encrypt_block(data)
    encrypt_time = time.time() - start_time
    
    logger.info(f"Время шифрования: {encrypt_time:.4f} секунд")
    logger.info(f"Скорость шифрования: {len(data) / encrypt_time:.2f} байт/сек")
    
    # Расшифрование
    start_time = time.time()
    decrypted = hfe_instance.decrypt_block(encrypted)
    decrypt_time = time.time() - start_time
    
    logger.info(f"Время расшифрования: {decrypt_time:.4f} секунд")
    logger.info(f"Скорость расшифрования: {len(data) / decrypt_time:.2f} байт/сек")
    
    # Проверка корректности
    success = data == decrypted
    if success:
        logger.info("✓ Расшифрование успешно: данные совпадают с оригиналом")
    else:
        logger.error("✗ Ошибка: расшифрованные данные не совпадают с оригиналом")
        logger.debug(f"Оригинал: {data[:20]}...")
        logger.debug(f"Расшифровано: {decrypted[:20]}...")
    
    total_time = encrypt_time + decrypt_time
    logger.info(f"Общее время: {total_time:.4f} секунд")
    logger.info("=" * 50)
    
    return encrypt_time, decrypt_time, success


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="HFE криптосистема с поддержкой параллельных вычислений"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["base", "cpu", "gpu", "all"],
        default="all",
        help="Режим работы: base (обычный), cpu (CPU-параллельный), gpu (GPU-параллельный), all (все)"
    )
    parser.add_argument(
        "--data-size",
        type=int,
        default=1024,
        help="Размер тестовых данных в байтах (по умолчанию: 1024)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Размерность поля GF(2^n) (по умолчанию: 8)"
    )
    parser.add_argument(
        "--d",
        type=int,
        default=3,
        help="Степень HFE многочлена (по умолчанию: 3)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed для генерации ключей (по умолчанию: 42)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Уровень логирования (по умолчанию: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Путь к файлу для записи логов (опционально)"
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Количество процессов для CPU-параллельного режима (по умолчанию: количество ядер)"
    )
    
    args = parser.parse_args()
    
    # Настройка логирования
    global logger
    logger = setup_logger("HFE_Main", log_level=args.log_level, log_file=args.log_file)
    
    logger.info("=" * 60)
    logger.info("HFE Криптосистема")
    logger.info("=" * 60)
    logger.info(f"Параметры: n={args.n}, d={args.d}, seed={args.seed}")
    logger.info(f"Размер тестовых данных: {args.data_size} байт")
    logger.info(f"Режим работы: {args.mode}")
    
    # Генерация тестовых данных
    import random
    random.seed(args.seed)
    test_data = bytes([random.randint(0, 255) for _ in range(args.data_size)])
    logger.info(f"Тестовые данные сгенерированы: {test_data[:20].hex()}...")
    
    results = {}
    
    # Обычная версия
    if args.mode in ["base", "all"]:
        try:
            logger.info("\n" + "=" * 60)
            logger.info("ЗАПУСК ОБЫЧНОЙ ВЕРСИИ (без распараллеливания)")
            logger.info("=" * 60)
            hfe_base = HFEBase(n=args.n, d=args.d, seed=args.seed)
            encrypt_time, decrypt_time, success = benchmark_encryption_decryption(
                hfe_base, test_data, "Обычная версия"
            )
            results["base"] = {
                "encrypt_time": encrypt_time,
                "decrypt_time": decrypt_time,
                "total_time": encrypt_time + decrypt_time,
                "success": success
            }
        except Exception as e:
            logger.error(f"Ошибка в обычной версии: {e}", exc_info=True)
            results["base"] = {"error": str(e)}
    
    # CPU-параллельная версия
    if args.mode in ["cpu", "all"]:
        try:
            logger.info("\n" + "=" * 60)
            logger.info("ЗАПУСК CPU-ПАРАЛЛЕЛЬНОЙ ВЕРСИИ")
            logger.info("=" * 60)
            hfe_cpu = HFECPUParallel(
                n=args.n, 
                d=args.d, 
                seed=args.seed,
                num_processes=args.num_processes
            )
            encrypt_time, decrypt_time, success = benchmark_encryption_decryption(
                hfe_cpu, test_data, "CPU-параллельная версия"
            )
            results["cpu"] = {
                "encrypt_time": encrypt_time,
                "decrypt_time": decrypt_time,
                "total_time": encrypt_time + decrypt_time,
                "success": success
            }
        except Exception as e:
            logger.error(f"Ошибка в CPU-параллельной версии: {e}", exc_info=True)
            results["cpu"] = {"error": str(e)}
    
    # GPU-параллельная версия
    if args.mode in ["gpu", "all"]:
        if not CUDA_AVAILABLE:
            logger.warning("GPU-параллельная версия недоступна")
            logger.warning("Причина: CUDA драйвер не найден или NVIDIA GPU недоступен")
            logger.warning("Решение: Используйте --mode cpu для CPU-параллельной версии")
            logger.info("Подробности: см. CUDA_TROUBLESHOOTING.md")
            results["gpu"] = {"error": "CUDA не доступен"}
        else:
            try:
                logger.info("\n" + "=" * 60)
                logger.info("ЗАПУСК GPU-ПАРАЛЛЕЛЬНОЙ ВЕРСИИ")
                logger.info("=" * 60)
                hfe_gpu = HFEGPUParallel(n=args.n, d=args.d, seed=args.seed)
                encrypt_time, decrypt_time, success = benchmark_encryption_decryption(
                    hfe_gpu, test_data, "GPU-параллельная версия"
                )
                results["gpu"] = {
                    "encrypt_time": encrypt_time,
                    "decrypt_time": decrypt_time,
                    "total_time": encrypt_time + decrypt_time,
                    "success": success
                }
            except Exception as e:
                logger.error(f"Ошибка в GPU-параллельной версии: {e}", exc_info=True)
                results["gpu"] = {"error": str(e)}
    
    # Сравнение результатов
    if len(results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
        logger.info("=" * 60)
        
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        
        if len(valid_results) > 1:
            # Находим базовое время для сравнения
            base_time = valid_results.get("base", {}).get("total_time")
            
            for mode, result in valid_results.items():
                total_time = result.get("total_time", 0)
                success = result.get("success", False)
                status = "✓" if success else "✗"
                
                if base_time and mode != "base":
                    speedup = base_time / total_time if total_time > 0 else 0
                    logger.info(
                        f"{status} {mode.upper():15s}: {total_time:.4f} сек "
                        f"(ускорение: {speedup:.2f}x)"
                    )
                else:
                    logger.info(f"{status} {mode.upper():15s}: {total_time:.4f} сек")
        else:
            logger.warning("Недостаточно результатов для сравнения")
    
    logger.info("\n" + "=" * 60)
    logger.info("РАБОТА ЗАВЕРШЕНА")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

