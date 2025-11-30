#!/bin/bash
# Тест базовой версии HFE (без распараллеливания)

echo "=========================================="
echo "Тест базовой версии HFE"
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

echo "Запуск базовой версии с размером данных 2048 байт..."
python3 main.py --mode base --data-size 2048 --log-level INFO

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Тест базовой версии пройден успешно"
    exit 0
else
    echo ""
    echo "✗ Тест базовой версии провален"
    exit 1
fi

