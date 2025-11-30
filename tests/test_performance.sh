#!/bin/bash
# Тест производительности всех версий HFE

echo "=========================================="
echo "Тест производительности HFE"
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

# Размеры данных для тестирования
SIZES=(512 1024 2048 4096 8192)

echo "Тестирование производительности с разными размерами данных..."
echo ""

# Создание директории для результатов
mkdir -p logs
RESULTS_FILE="logs/performance_results.log"

# Очистка файла результатов
echo "Результаты тестирования производительности" > "$RESULTS_FILE"
echo "Дата: $(date)" >> "$RESULTS_FILE"
echo "==========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for size in "${SIZES[@]}"; do
    echo "Тестирование с размером данных: ${size} байт"
    echo "Размер данных: ${size} байт" >> "$RESULTS_FILE"
    echo "----------------------------------------" >> "$RESULTS_FILE"
    
    # Запуск теста
    python3 main.py --mode all --data-size "$size" --log-level INFO >> "$RESULTS_FILE" 2>&1
    
    echo "" >> "$RESULTS_FILE"
    echo ""
done

echo ""
echo "=========================================="
echo "Тестирование завершено"
echo "Результаты сохранены в: $RESULTS_FILE"
echo "=========================================="

