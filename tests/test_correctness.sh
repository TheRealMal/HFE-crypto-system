#!/bin/bash
# Тест корректности работы HFE (проверка шифрование/расшифрование)

echo "=========================================="
echo "Тест корректности работы HFE"
echo "=========================================="

cd "$(dirname "$0")/.." || exit 1

# Проверка наличия Python
if ! command -v python3 &> /dev/null; then
    echo "ОШИБКА: python3 не найден"
    exit 1
fi

# Создание тестового скрипта Python
PROJECT_ROOT=$(pwd)
cat > /tmp/test_hfe_correctness.py << PYTHON_EOF
import sys
import os
from pathlib import Path

# Добавление корневой директории проекта в путь
project_root = Path("$PROJECT_ROOT")
sys.path.insert(0, str(project_root))

from src.hfe_base import HFEBase

# Тестовые данные
test_cases = [
    b"Hello, World!",
    b"Test data 123",
    b"",
    bytes(range(256)),  # Все возможные байты
    b"A" * 100,  # Повторяющиеся символы
]

print("Тестирование базовой версии HFE...")
hfe_base = HFEBase(n=8, d=3, seed=42)
for i, data in enumerate(test_cases):
    try:
        encrypted = hfe_base.encrypt_block(data)
        decrypted = hfe_base.decrypt_block(encrypted)
        if data == decrypted:
            print(f"  ✓ Тест {i+1} пройден (размер: {len(data)} байт)")
        else:
            print(f"  ✗ Тест {i+1} провален (размер: {len(data)} байт)")
            print(f"     Ожидалось: {data[:20]}...")
            print(f"     Получено:  {decrypted[:20]}...")
            sys.exit(1)
    except Exception as e:
        print(f"  ✗ Тест {i+1} вызвал ошибку: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

print("\n✓ Все тесты корректности базовой версии пройдены успешно!")
print("Примечание: CPU и GPU версии тестируются через main.py")
PYTHON_EOF

python3 /tmp/test_hfe_correctness.py

RESULT=$?
rm -f /tmp/test_hfe_correctness.py

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Тест корректности пройден успешно"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "✗ Тест корректности провален"
    echo "=========================================="
    exit 1
fi

