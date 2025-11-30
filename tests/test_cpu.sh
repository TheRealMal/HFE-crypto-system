#!/bin/bash
# Тест CPU-параллельной версии HFE

echo "=========================================="
echo "Тест CPU-параллельной версии HFE"
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

echo "Запуск CPU-параллельной версии с размером данных 2048 байт..."
echo "Использование 4 процессов..."
python3 main.py --mode cpu --data-size 2048 --num-processes 4 --log-level INFO

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Тест CPU-параллельной версии пройден успешно"
    exit 0
else
    echo ""
    echo "✗ Тест CPU-параллельной версии провален"
    exit 1
fi

