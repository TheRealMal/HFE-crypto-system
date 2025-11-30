# Устранение проблем с CUDA

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

1. **Установите CUDA Toolkit**:
   - Linux: Следуйте инструкциям на [nvidia.com/cuda](https://developer.nvidia.com/cuda-downloads)
   - Windows: Загрузите установщик с официального сайта NVIDIA

2. **Установите Numba с поддержкой CUDA**:
   ```bash
   pip install numba
   ```

3. **Проверьте доступность CUDA**:
   ```bash
   python3 -c "from numba import cuda; print('CUDA available:', cuda.is_available())"
   ```

### Решение 3: Установить переменную окружения (Если драйвер установлен, но не найден)

Если CUDA драйвер установлен, но Numba его не находит:

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
- Откройте Диспетчер устройств → Видеоадаптеры

### Проверка установки CUDA

```bash
nvcc --version
nvidia-smi
```

### Проверка Numba

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

### Для Linux/Windows с NVIDIA GPU

1. Убедитесь, что у вас установлен NVIDIA драйвер
2. Установите CUDA Toolkit
3. Установите Numba: `pip install numba`
4. Проверьте доступность: `python -c "from numba import cuda; print(cuda.is_available())"`

## Дополнительная информация

- [Документация Numba CUDA](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
- [Numba CUDA Installation Guide](https://numba.readthedocs.io/en/stable/cuda/overview.html#requirements)

## Важно

**Ошибка CUDA не является критической** - программа продолжит работать с базовой и CPU-параллельной версиями. GPU-версия является опциональной и используется только при наличии подходящего оборудования.

