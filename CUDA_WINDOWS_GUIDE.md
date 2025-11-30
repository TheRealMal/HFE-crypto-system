# Руководство по установке CUDA для Numba на Windows 10

## Пошаговая инструкция

### Шаг 1: Проверка системы

1. **Проверьте наличие NVIDIA GPU**:
   ```powershell
   # В PowerShell
   Get-WmiObject Win32_VideoController | Select-Object Name
   ```
   Или через Диспетчер устройств: Win+X → Диспетчер устройств → Видеоадаптеры

2. **Проверьте разрядность системы**:
   - Должна быть 64-битная Windows 10
   - 32-битные версии не поддерживают CUDA

3. **Проверьте разрядность Python**:
   ```cmd
   python -c "import platform; print(platform.architecture())"
   ```
   Должно быть `('64bit', 'WindowsPE')`

### Шаг 2: Установка драйверов NVIDIA

1. **Определите модель GPU**:
   - Через Диспетчер устройств
   - Или через `nvidia-smi` (если уже установлены драйверы)

2. **Скачайте драйверы**:
   - Перейдите на [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)
   - Выберите вашу модель GPU и Windows 10
   - Скачайте и установите последние драйверы

3. **Проверьте установку**:
   ```cmd
   nvidia-smi
   ```
   Должна отобразиться информация о GPU и версии драйвера

### Шаг 3: Установка CUDA Toolkit

1. **Определите совместимую версию CUDA**:
   - Проверьте через `nvidia-smi` (показывает максимальную поддерживаемую версию CUDA)
   - Или используйте последнюю стабильную версию (например, CUDA 11.8 или 12.x)

2. **Скачайте CUDA Toolkit**:
   - Перейдите на [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Выберите:
     - Operating System: Windows
     - Architecture: x86_64
     - Version: Windows 10
     - Installer Type: exe (local)

3. **Установите CUDA Toolkit**:
   - Запустите установщик
   - **Важно**: Выберите "Custom installation" и убедитесь, что установлены:
     - CUDA Development Tools
     - CUDA Samples (опционально, для тестирования)
   - Запишите путь установки (обычно `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X`)

4. **Проверьте установку**:
   ```cmd
   nvcc --version
   ```
   Должна отобразиться версия CUDA компилятора

### Шаг 4: Настройка переменных окружения

1. **Откройте настройки переменных окружения**:
   - Win+R → введите `sysdm.cpl` → Enter
   - Вкладка "Дополнительно" → "Переменные среды"

2. **Добавьте/измените переменную CUDA_HOME**:
   - В разделе "Переменные пользователя" или "Системные переменные"
   - Нажмите "Создать" (если нет) или "Изменить" (если есть)
   - Имя переменной: `CUDA_HOME`
   - Значение: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`
     (замените v11.8 на вашу версию)

3. **Добавьте пути в PATH**:
   - Найдите переменную `Path` в том же списке
   - Нажмите "Изменить" → "Создать"
   - Добавьте следующие пути (замените версию на свою):
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\CUPTI\lib64
     ```

4. **Примените изменения**:
   - Нажмите "ОК" во всех окнах
   - **Перезапустите все открытые терминалы/IDE** (важно!)

### Шаг 5: Установка Numba

#### Вариант A: Через pip (простой способ)

```cmd
pip install numba
```

#### Вариант B: Через conda (рекомендуется для Windows)

Если у вас установлен Anaconda или Miniconda:

```cmd
conda install numba
conda install cudatoolkit
```

**Преимущества conda:**
- Автоматически устанавливает совместимые версии
- Включает необходимые библиотеки CUDA
- Меньше проблем с зависимостями

### Шаг 6: Проверка установки

1. **Проверьте версию Numba**:
   ```cmd
   python -c "import numba; print('Numba version:', numba.__version__)"
   ```

2. **Проверьте доступность CUDA**:
   ```cmd
   python -c "from numba import cuda; print('CUDA available:', cuda.is_available())"
   ```

3. **Детальная диагностика**:
   ```python
   from numba import cuda
   print("CUDA available:", cuda.is_available())
   if cuda.is_available():
       print("CUDA devices:", len(cuda.gpus))
       for i, device in enumerate(cuda.gpus):
           print(f"Device {i}: {device}")
   else:
       print("CUDA not available. Check installation.")
   ```

### Шаг 7: Решение проблем

#### Проблема: "CUDA driver library cannot be found"

**Решение 1: Установите переменную NUMBA_CUDA_DRIVER**

В PowerShell (от имени администратора):
```powershell
[System.Environment]::SetEnvironmentVariable("NUMBA_CUDA_DRIVER", "C:\Windows\System32\nvcuda.dll", "User")
```

Или через GUI:
- Win+R → `sysdm.cpl` → Переменные среды
- Создайте переменную:
  - Имя: `NUMBA_CUDA_DRIVER`
  - Значение: `C:\Windows\System32\nvcuda.dll`

**Решение 2: Проверьте PATH**

Убедитесь, что пути к CUDA добавлены в PATH:
```cmd
echo %PATH% | findstr CUDA
```

**Решение 3: Используйте conda**

Если pip не работает, попробуйте conda:
```cmd
conda install -c conda-forge numba cudatoolkit
```

#### Проблема: "Numba requires 64-bit Python"

**Решение:**
- Установите 64-битную версию Python
- Удалите 32-битную версию, если установлена

#### Проблема: Версии не совместимы

**Решение:**
- Используйте conda для автоматической совместимости
- Или установите совместимые версии вручную:
  - CUDA Toolkit 11.x → Numba 0.56+
  - CUDA Toolkit 12.x → Numba 0.57+

#### Проблема: Антивирус блокирует

**Решение:**
- Добавьте исключения для:
  - `C:\Program Files\NVIDIA GPU Computing Toolkit\`
  - `C:\Windows\System32\nvcuda.dll`
  - Python и pip исполняемые файлы

### Шаг 8: Тестирование

После успешной установки протестируйте:

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_arrays(a, b, c):
    idx = cuda.grid(1)
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

# Тест
if cuda.is_available():
    print("CUDA работает!")
    # Создаем тестовые данные
    n = 1000
    a = np.ones(n)
    b = np.ones(n)
    c = np.zeros(n)
    
    # Копируем на GPU
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    d_c = cuda.to_device(c)
    
    # Запускаем kernel
    add_arrays[1, n](d_a, d_b, d_c)
    
    # Копируем обратно
    result = d_c.copy_to_host()
    print("Тест пройден!" if np.allclose(result, 2.0) else "Ошибка!")
else:
    print("CUDA недоступна")
```

## Быстрая проверка

Выполните все команды последовательно:

```cmd
# 1. Проверка GPU
nvidia-smi

# 2. Проверка CUDA
nvcc --version

# 3. Проверка переменных
echo %CUDA_HOME%
echo %PATH% | findstr CUDA

# 4. Проверка Python
python -c "import platform; print(platform.architecture())"

# 5. Проверка Numba
python -c "from numba import cuda; print('CUDA:', cuda.is_available())"
```

## Альтернативное решение: Использовать CPU-версию

Если установка CUDA вызывает проблемы, используйте CPU-параллельную версию:

```cmd
python main.py --mode cpu
```

Она работает без CUDA и обеспечивает хорошую производительность на многоядерных CPU.

## Полезные ссылки

- [Официальная документация Numba CUDA](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- [Numba Installation Guide](https://numba.readthedocs.io/en/stable/user/installing.html)
- [CUDA Compatibility Guide](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)

## Контакты для помощи

Если проблема не решается:
1. Проверьте [Issues на GitHub Numba](https://github.com/numba/numba/issues)
2. Посетите [форум Numba](https://numba.discourse.group/)
3. Проверьте логи ошибок для детальной информации

