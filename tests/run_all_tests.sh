#!/bin/bash
# Запуск всех тестов

echo "=========================================="
echo "Запуск всех тестов HFE"
echo "=========================================="
echo ""

# Цвета для вывода
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Счетчики
PASSED=0
FAILED=0
SKIPPED=0

# Функция для запуска теста
run_test() {
    local test_name=$1
    local test_script=$2
    
    echo "----------------------------------------"
    echo "Запуск: $test_name"
    echo "----------------------------------------"
    
    if [ ! -f "$test_script" ]; then
        echo -e "${YELLOW}⚠ Тест не найден: $test_script${NC}"
        ((SKIPPED++))
        return
    fi
    
    if bash "$test_script"; then
        echo -e "${GREEN}✓ $test_name пройден${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ $test_name провален${NC}"
        ((FAILED++))
    fi
    echo ""
}

# Переход в директорию скрипта
cd "$(dirname "$0")/.." || exit 1

# Запуск тестов
run_test "Тест корректности" "tests/test_correctness.sh"
run_test "Тест базовой версии" "tests/test_base.sh"
run_test "Тест CPU-параллельной версии" "tests/test_cpu.sh"
run_test "Тест GPU-параллельной версии" "tests/test_gpu.sh"
run_test "Комплексный тест всех версий" "tests/test_all.sh"

# Итоги
echo "=========================================="
echo "Итоги тестирования"
echo "=========================================="
echo -e "${GREEN}Пройдено: $PASSED${NC}"
echo -e "${RED}Провалено: $FAILED${NC}"
echo -e "${YELLOW}Пропущено: $SKIPPED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ Все тесты пройдены успешно!${NC}"
    exit 0
else
    echo -e "${RED}✗ Некоторые тесты провалены${NC}"
    exit 1
fi

