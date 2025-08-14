"""
Утилиты для настройки логирования проекта.
"""

import os
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str, 
                 level: str = "INFO",
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> logging.Logger:
    """
    Настройка логгера для проекта.
    
    Args:
        name: Имя логгера
        level: Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Путь к файлу для записи логов (опционально)
        format_string: Формат логов (опционально)
        
    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    
    # Избегаем дублирования handlers при повторном вызове
    if logger.handlers:
        return logger
    
    # Устанавливаем уровень
    logger.setLevel(getattr(logging, level.upper()))
    
    # Формат по умолчанию
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (если указан файл)
    if log_file:
        # Создаем директорию для логов если она не существует
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_project_logging(config: dict) -> None:
    """
    Настройка логирования для всего проекта на основе конфигурации.
    
    Args:
        config: Конфигурация проекта с секцией logging
    """
    logging_config = config.get('logging', {})
    
    level = logging_config.get('level', 'INFO')
    format_string = logging_config.get('format', 
                                     "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_file = logging_config.get('log_file', None)
    
    # Настройка root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Очищаем существующие handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Получение логгера по имени.
    
    Args:
        name: Имя логгера
        
    Returns:
        Логгер
    """
    return logging.getLogger(name)


def log_function_call(func):
    """
    Декоратор для логирования вызовов функций.
    
    Args:
        func: Функция для декорирования
        
    Returns:
        Декорированная функция
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Вызов функции {func.__name__} с args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Функция {func.__name__} завершилась успешно")
            return result
        except Exception as e:
            logger.error(f"Ошибка в функции {func.__name__}: {e}")
            raise
    
    return wrapper


def configure_external_loggers(level: str = "WARNING"):
    """
    Настройка уровня логирования для внешних библиотек.
    
    Args:
        level: Уровень логирования для внешних библиотек
    """
    external_loggers = [
        'urllib3',
        'requests',
        'matplotlib',
        'PIL',
        'asyncio',
        'mlflow',
        'catboost',
        'xgboost'
    ]
    
    for logger_name in external_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))