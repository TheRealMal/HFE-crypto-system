"""
Конфигурация логирования для проекта HFE
"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str, log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """
    Настройка логгера с поддержкой разных уровней логирования
    
    Args:
        name: Имя логгера
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу для записи логов (опционально)
    
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Очистка существующих обработчиков
    logger.handlers.clear()
    
    # Формат логов
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Консольный обработчик с безопасной обработкой Unicode для Windows
    class SafeStreamHandler(logging.StreamHandler):
        """Обработчик потока с безопасной обработкой Unicode для Windows"""
        def emit(self, record):
            try:
                super().emit(record)
            except UnicodeEncodeError:
                # Если не удается закодировать, заменяем проблемные символы
                try:
                    msg = self.format(record)
                    # Заменяем проблемные Unicode символы на ASCII
                    safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
                    stream = self.stream
                    stream.write(safe_msg + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)
    
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Файловый обработчик (если указан)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

