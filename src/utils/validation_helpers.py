"""
Простые утилиты для валидации входных данных в функциях обработки данных.
"""

import pandas as pd
import numpy as np
from typing import Any, List, Optional, Union
import os
from pathlib import Path


def validate_dataframe_input(df: Any, name: str = "DataFrame") -> pd.DataFrame:
    """
    Валидация входного DataFrame
    
    Args:
        df: Объект для проверки
        name: Имя DataFrame для сообщений об ошибках
        
    Returns:
        Проверенный DataFrame
        
    Raises:
        TypeError: Если входные данные не DataFrame
        ValueError: Если DataFrame пустой
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} должен быть pandas.DataFrame, получен {type(df)}")
    
    if df.empty:
        raise ValueError(f"{name} не может быть пустым")
    
    return df


def validate_config_input(config: Any, required_sections: Optional[List[str]] = None) -> dict:
    """
    Валидация конфигурации
    
    Args:
        config: Объект конфигурации для проверки
        required_sections: Обязательные секции
        
    Returns:
        Проверенная конфигурация
        
    Raises:
        TypeError: Если config не словарь
        ValueError: Если отсутствуют обязательные секции
    """
    if not isinstance(config, dict):
        raise TypeError(f"config должен быть словарем, получен {type(config)}")
    
    if required_sections:
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            raise ValueError(f"Отсутствуют обязательные секции в конфигурации: {missing_sections}")
    
    return config


def validate_file_path(file_path: Union[str, Path], must_exist: bool = True, 
                      must_be_readable: bool = True) -> Path:
    """
    Валидация пути к файлу
    
    Args:
        file_path: Путь к файлу
        must_exist: Файл должен существовать
        must_be_readable: Файл должен быть доступен для чтения
        
    Returns:
        Проверенный путь как Path объект
        
    Raises:
        TypeError: Если file_path неверного типа
        FileNotFoundError: Если файл не существует (при must_exist=True)
        PermissionError: Если файл недоступен для чтения
    """
    if not isinstance(file_path, (str, Path)):
        raise TypeError(f"file_path должен быть str или Path, получен {type(file_path)}")
    
    path_obj = Path(file_path)
    
    if must_exist and not path_obj.exists():
        raise FileNotFoundError(f"Файл не найден: {path_obj}")
    
    if must_exist and must_be_readable and not os.access(path_obj, os.R_OK):
        raise PermissionError(f"Файл недоступен для чтения: {path_obj}")
    
    return path_obj


def validate_columns_exist(df: pd.DataFrame, required_columns: List[str], 
                          df_name: str = "DataFrame") -> None:
    """
    Проверка наличия обязательных столбцов в DataFrame
    
    Args:
        df: DataFrame для проверки
        required_columns: Список обязательных столбцов
        df_name: Имя DataFrame для сообщений об ошибках
        
    Raises:
        ValueError: Если отсутствуют обязательные столбцы
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"В {df_name} отсутствуют обязательные столбцы: {missing_columns}")


def validate_numeric_range(value: Union[int, float], min_val: Optional[float] = None, 
                          max_val: Optional[float] = None, name: str = "value") -> None:
    """
    Валидация числового значения в диапазоне
    
    Args:
        value: Значение для проверки
        min_val: Минимальное значение (включительно)
        max_val: Максимальное значение (включительно)
        name: Имя параметра для сообщений об ошибках
        
    Raises:
        TypeError: Если value не число
        ValueError: Если value вне допустимого диапазона
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} должен быть числом, получен {type(value)}")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{name} ({value}) должен быть >= {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{name} ({value}) должен быть <= {max_val}")


def validate_data_quality(df: pd.DataFrame, 
                         max_missing_ratio: float = 0.95,
                         check_inf: bool = True,
                         check_duplicates: bool = False,
                         unique_columns: Optional[List[str]] = None) -> dict:
    """
    Комплексная проверка качества данных
    
    Args:
        df: DataFrame для проверки
        max_missing_ratio: Максимальная доля пропусков в столбце
        check_inf: Проверять бесконечные значения
        check_duplicates: Проверять дубликаты
        unique_columns: Столбцы для проверки уникальности
        
    Returns:
        Словарь с результатами проверки
    """
    results = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Проверка пропущенных значений
    missing_ratios = df.isnull().sum() / len(df)
    high_missing = missing_ratios[missing_ratios > max_missing_ratio]
    if not high_missing.empty:
        results['warnings'].append(f"Столбцы с высокой долей пропусков: {high_missing.to_dict()}")
    
    # Проверка бесконечных значений
    if check_inf:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = []
        for col in numeric_cols:
            if df[col].isin([np.inf, -np.inf]).any():
                inf_cols.append(col)
        
        if inf_cols:
            results['errors'].append(f"Столбцы с бесконечными значениями: {inf_cols}")
            results['is_valid'] = False
    
    # Проверка дубликатов
    if check_duplicates:
        if df.duplicated().any():
            dup_count = df.duplicated().sum()
            results['warnings'].append(f"Найдено {dup_count} дублированных строк")
    
    # Проверка уникальности определенных столбцов
    if unique_columns:
        for col in unique_columns:
            if col in df.columns and df[col].duplicated().any():
                dup_count = df[col].duplicated().sum()
                results['warnings'].append(f"Столбец '{col}' содержит {dup_count} дубликатов")
    
    return results


def safe_create_directory(directory_path: Union[str, Path]) -> Path:
    """
    Безопасное создание директории с проверками
    
    Args:
        directory_path: Путь к директории
        
    Returns:
        Path объект созданной директории
        
    Raises:
        TypeError: Если directory_path неверного типа
        RuntimeError: Если не удалось создать директорию
    """
    if not isinstance(directory_path, (str, Path)):
        raise TypeError(f"directory_path должен быть str или Path, получен {type(directory_path)}")
    
    path_obj = Path(directory_path)
    
    try:
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    except OSError as e:
        raise RuntimeError(f"Не удалось создать директорию {path_obj}: {e}")