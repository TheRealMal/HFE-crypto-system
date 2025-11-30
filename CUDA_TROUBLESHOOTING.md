# Устранение проблем с CUDA

> **Для пользователей Windows 10**: См. подробное руководство [CUDA_WINDOWS_GUIDE.md](CUDA_WINDOWS_GUIDE.md)

## Проблема: "CUDA driver library cannot be found"

Эта ошибка возникает, когда Numba установлен, но CUDA драйвер недоступен или не может быть найден.

### Причины

1. **macOS без NVIDIA GPU**: На Mac с Apple Silicon (M1/M2/M3) или Intel без NVIDIA GPU CUDA недоступен, так как CUDA работает только с NVIDIA GPU.

2. **CUDA Toolkit не установлен**: Numba установлен, но CUDA Toolkit отсутствует.

3. **Неправильная версия CUDA**: Версия CUDA Toolkit не совместима с установленным драйвером.

4. **Драйвер не найден**: Система не может найти CUDA драйвер библиотеку.

## Решения

### Решение 1: Использовать CPU-параллельную версию (Рекомендуется для macOS)

Если у вас нет NVIDIA GPU (например, на macOS), используйте CPU-параллельную версию:

```bash
python main.py --mode cpu
# или
python main.py --mode all  # GPU версия будет пропущена с предупреждением
```

### Решение 2: Установить CUDA Toolkit (Только для систем с NVIDIA GPU)

Если у вас есть NVIDIA GPU:

#### Для Windows 10:

1. **Проверьте наличие NVIDIA GPU**:
   - Откройте Диспетчер устройств (Win+X → Диспетчер устройств)
   - Раздел "Видеоадаптеры" → должна быть видеокарта NVIDIA
   - Если нет, CUDA недоступен

2. **Установите/обновите драйверы NVIDIA**:
   - Скачайте последние драйверы с [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx)
   - Установите драйверы и перезагрузите компьютер
   - Проверьте: `nvidia-smi` в командной строке (должна показать информацию о GPU)

3. **Установите CUDA Toolkit**:
   - Скачайте CUDA Toolkit с [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Выберите версию, совместимую с вашим драйвером (проверьте через `nvidia-smi`)
   - Установите CUDA Toolkit (обычно в `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X`)
   - **Важно**: Убедитесь, что установлены все компоненты, включая "CUDA Development Tools"

4. **Настройте переменные окружения**:
   - Откройте "Переменные среды" (Win+R → `sysdm.cpl` → Дополнительно → Переменные среды)
   - Добавьте/проверьте переменную `CUDA_HOME`:
     ```
     CUDA_HOME = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
     ```
     (замените v11.8 на вашу версию)
   - Добавьте в `PATH`:
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
     ```
   - Перезапустите командную строку/PowerShell после изменений

5. **Установите Numba**:
   ```bash
   pip install numba
   ```
   
   **Альтернатива через conda (рекомендуется для Windows)**:
   ```bash
   conda install numba
   conda install cudatoolkit
   ```

6. **Проверьте доступность CUDA**:
   ```bash
   python -c "from numba import cuda; print('CUDA available:', cuda.is_available())"
   ```

#### Для Linux:

1. **Установите CUDA Toolkit**:
   - Следуйте инструкциям на [nvidia.com/cuda](https://developer.nvidia.com/cuda-downloads)
   - Или через пакетный менеджер вашего дистрибутива

2. **Установите Numba**:
   ```bash
   pip install numba
   ```

3. **Проверьте доступность CUDA**:
   ```bash
   python3 -c "from numba import cuda; print('CUDA available:', cuda.is_available())"
   ```

### Решение 3: Установить переменную окружения (Если драйвер установлен, но не найден)

Если CUDA драйвер установлен, но Numba его не находит:

#### Для Windows 10:

1. **Найдите путь к CUDA драйверу**:
   - Обычно: `C:\Windows\System32\nvcuda.dll`
   - Или: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin\cudart64_XX.dll`

2. **Установите переменную окружения**:
   - Откройте PowerShell или CMD от имени администратора
   - Выполните:
     ```powershell
     # Временная установка (только для текущей сессии)
     $env:NUMBA_CUDA_DRIVER = "C:\Windows\System32\nvcuda.dll"
     
     # Или постоянная установка
     [System.Environment]::SetEnvironmentVariable("NUMBA_CUDA_DRIVER", "C:\Windows\System32\nvcuda.dll", "User")
     ```
   - Перезапустите терминал

3. **Альтернативно через GUI**:
   - Win+R → `sysdm.cpl` → Дополнительно → Переменные среды
   - Добавьте новую переменную:
     - Имя: `NUMBA_CUDA_DRIVER`
     - Значение: `C:\Windows\System32\nvcuda.dll`

#### Для Linux:

```bash
# Linux
export NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so

# Или укажите путь к библиотеке драйвера
export NUMBA_CUDA_DRIVER=/path/to/libcuda.so
```

Затем запустите программу:
```bash
python main.py --mode gpu
```

### Решение 4: Использовать виртуальную машину или удаленный сервер

Если у вас нет NVIDIA GPU локально, но нужна GPU-версия:

1. Используйте облачный сервис (Google Colab, AWS EC2 с GPU, Azure)
2. Используйте удаленный сервер с NVIDIA GPU
3. Используйте виртуальную машину с GPU passthrough

## Проверка системы

### Проверка наличия NVIDIA GPU

**Linux:**
```bash
lspci | grep -i nvidia
nvidia-smi
```

**macOS:**
```bash
system_profiler SPDisplaysDataType | grep -i nvidia
```
(Обычно на macOS NVIDIA GPU отсутствует)

**Windows:**
- Откройте Диспетчер устройств (Win+X → Диспетчер устройств) → Видеоадаптеры
- Или выполните в PowerShell:
  ```powershell
  Get-WmiObject Win32_VideoController | Select-Object Name, DriverVersion
  ```
- Проверьте через `nvidia-smi` в командной строке (если установлены драйверы)

### Проверка установки CUDA

**Windows:**
```cmd
# Проверка версии CUDA компилятора
nvcc --version

# Проверка драйверов и GPU
nvidia-smi

# Если команды не найдены, проверьте PATH
echo %CUDA_HOME%
```

**Linux:**
```bash
nvcc --version
nvidia-smi
```

### Проверка Numba

**Windows:**
```cmd
python -c "import numba; print(numba.__version__)"
python -c "from numba import cuda; print('CUDA available:', cuda.is_available())"

# Детальная диагностика
python -c "from numba import cuda; print('CUDA devices:', cuda.gpus)"
```

**Linux:**
```bash
python3 -c "import numba; print(numba.__version__)"
python3 -c "from numba import cuda; print('CUDA available:', cuda.is_available())"
```

## Рекомендации

### Для macOS пользователей

**Используйте CPU-параллельную версию** - это оптимальный вариант, так как:
- macOS не поддерживает NVIDIA GPU (кроме старых моделей)
- CPU-параллелизация работает отлично на многоядерных процессорах
- Не требует дополнительных драйверов

```bash
python main.py --mode cpu --num-processes 8
```

### Для Windows 10 с NVIDIA GPU

1. **Убедитесь, что у вас 64-битная система** (32-битные версии не поддерживают CUDA)
2. **Установите/обновите NVIDIA драйверы** через официальный сайт NVIDIA
3. **Установите CUDA Toolkit** (совместимую версию с вашим драйвером)
4. **Настройте переменные окружения** (`CUDA_HOME` и `PATH`)
5. **Установите Numba**: 
   - Через pip: `pip install numba`
   - Или через conda: `conda install numba cudatoolkit` (рекомендуется)
6. **Проверьте доступность**: `python -c "from numba import cuda; print(cuda.is_available())"`

**Частые проблемы на Windows:**
- CUDA Toolkit не добавлен в PATH → добавьте вручную
- Несовместимость версий → используйте conda для автоматической совместимости
- Антивирус блокирует → добавьте исключения для CUDA файлов
- Нужны права администратора → запустите терминал от имени администратора

### Для Linux с NVIDIA GPU

1. Убедитесь, что у вас установлен NVIDIA драйвер
2. Установите CUDA Toolkit
3. Установите Numba: `pip install numba`
4. Проверьте доступность: `python3 -c "from numba import cuda; print(cuda.is_available())"`

## Дополнительная информация

- [Документация Numba CUDA](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- [Numba CUDA Installation Guide](https://numba.readthedocs.io/en/stable/cuda/overview.html#requirements)

## Важно

**Ошибка CUDA не является критической** - программа продолжит работать с базовой и CPU-параллельной версиями. GPU-версия является опциональной и используется только при наличии подходящего оборудования.

