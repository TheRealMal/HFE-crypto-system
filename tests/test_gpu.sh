#!/bin/bash
# Тест GPU-параллельной версии HFE

echo "=========================================="
echo "Тест GPU-параллельной версии HFE"
echo "=========================================="

cd "$(dirname "$0")/.." || exit 1

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    echo "ОШИБКА: python3 не найден"
    exit 1
fi

# Проверка наличия main.py
if [ ! -f "main.py" ]; then
    echo "ОШИБКА: main.py не найден"
    exit 1
fi

# Проверка доступности CUDA
echo "Проверка доступности CUDA..."
python3 -c "from src.hfe_gpu_parallel import CUDA_AVAILABLE; exit(0 if CUDA_AVAILABLE else 1)" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "ПРЕДУПРЕЖДЕНИЕ: CUDA недоступен, тест будет пропущен"
    echo "Для использования GPU-версии установите: pip install numba"
    exit 0
fi

echo "Запуск GPU-параллельной версии с размером данных 65536 байт..."
python3 main.py --mode gpu --data-size 65536 --log-level INFO

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Тест GPU-параллельной версии пройден успешно"
    exit 0
else
    echo ""
    echo "✗ Тест GPU-параллельной версии провален"
    exit 1
fi

