#!/usr/bin/env python3
"""
Скрипт для предобработки данных.
Включает фильтрацию, разбиение на train/val/test и базовую очистку.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')
from utils.config import load_config
from utils.logging import setup_logger


class DataPreprocessor:
    """Класс для предобработки данных"""
    
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError("config должен быть словарем")
        
        self.config = config
        self.logger = setup_logger('DataPreprocessor')
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        required_sections = ['data', 'preprocessing']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Отсутствует секция '{section}' в конфигурации")
        
        required_data_fields = ['target_column', 'date_column']
        for field in required_data_fields:
            if field not in self.config['data']:
                raise ValueError(f"Отсутствует поле '{field}' в секции data")
    
    def _get_data_path(self, filename: str = "dataset.csv") -> Path:
        """Получение пути к файлу данных из конфигурации или по умолчанию"""
        # Проверяем конфигурацию на наличие базовых путей
        base_path = self.config.get('paths', {}).get('data_raw', 'data/raw')
        return Path(base_path) / filename
        
    def load_raw_data(self, file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """Загрузка исходных данных с проверками"""
        if file_path is None:
            data_path = self._get_data_path()
        else:
            data_path = Path(file_path)
        
        # Проверяем существование файла
        if not data_path.exists():
            raise FileNotFoundError(f"Файл данных не найден: {data_path}")
        
        # Проверяем что файл не пустой
        if data_path.stat().st_size == 0:
            raise ValueError(f"Файл данных пустой: {data_path}")
            
        try:
            self.logger.info(f"Загрузка данных из {data_path}")
            df = pd.read_csv(data_path)
            
            # Базовая валидация загруженных данных
            self._validate_loaded_data(df)
            
            self.logger.info(f"Размер исходных данных: {df.shape}")
            self.logger.info(f"Столбцы: {list(df.columns)}")
            
            return df
            
        except pd.errors.EmptyDataError:
            raise ValueError(f"Файл {data_path} не содержит данных")
        except pd.errors.ParserError as e:
            raise ValueError(f"Ошибка парсинга файла {data_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Неожиданная ошибка при загрузке {data_path}: {e}")
    
    def _validate_loaded_data(self, df: pd.DataFrame) -> None:
        """Валидация загруженных данных"""
        if df.empty:
            raise ValueError("Загруженный DataFrame пустой")
        
        # Проверяем наличие обязательных столбцов
        required_columns = [
            self.config['data']['target_column'],
            self.config['data']['date_column']
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing_columns}")
    
    def apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Применение фильтров к данным"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Входные данные должны быть pandas.DataFrame")
        
        if df.empty:
            self.logger.warning("Пустой DataFrame передан в apply_filters")
            return df
            
        initial_size = len(df)
        filters = self.config.get('data', {}).get('filters', {})
        
        if not filters:
            self.logger.info("Нет фильтров для применения")
            return df
        
        # Фильтр по дате
        if 'max_date' in filters:
            date_col = self.config['data']['date_column']
            if date_col not in df.columns:
                self.logger.warning(f"Столбец даты '{date_col}' не найден, пропускаем фильтр по дате")
            else:
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    max_date = pd.to_datetime(filters['max_date'])
                    df = df[df[date_col] <= max_date]
                    self.logger.info(f"Фильтр по дате <= {filters['max_date']}: {len(df)} строк")
                except (ValueError, TypeError) as e:
                    self.logger.error(f"Ошибка при применении фильтра по дате: {e}")
        
        # Фильтр по минимальной сумме disbursement
        if 'min_disbursement' in filters:
            if 'disbursement' not in df.columns:
                self.logger.warning("Столбец 'disbursement' не найден, пропускаем фильтр")
            else:
                min_disbursement = filters['min_disbursement']
                if not isinstance(min_disbursement, (int, float)):
                    self.logger.error(f"Неверный тип min_disbursement: {type(min_disbursement)}")
                else:
                    df = df[df['disbursement'] > min_disbursement]
                    self.logger.info(f"Фильтр по disbursement > {min_disbursement}: {len(df)} строк")
        
        # Фильтр по возрасту
        if 'min_age' in filters:
            if 'age' not in df.columns:
                self.logger.warning("Столбец 'age' не найден, пропускаем фильтр")
            else:
                min_age = filters['min_age']
                if not isinstance(min_age, (int, float)):
                    self.logger.error(f"Неверный тип min_age: {type(min_age)}")
                else:
                    df = df[df['age'] >= min_age]
                    self.logger.info(f"Фильтр по age >= {min_age}: {len(df)} строк")
        
        self.logger.info(f"Применены фильтры: {initial_size} -> {len(df)} строк")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обработка пропущенных значений"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Входные данные должны быть pandas.DataFrame")
            
        if df.empty:
            self.logger.warning("Пустой DataFrame передан в handle_missing_values")
            return df
            
        # Удаление столбцов с большим количеством пропусков
        nan_threshold = self.config.get('preprocessing', {}).get('nan_threshold', 0.95)
        if not isinstance(nan_threshold, (int, float)) or not 0 <= nan_threshold <= 1:
            self.logger.warning(f"Неверный nan_threshold: {nan_threshold}, используем 0.95")
            nan_threshold = 0.95
            
        missing_ratio = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratio[missing_ratio > nan_threshold].index.tolist()
        
        # Проверяем что не удаляем обязательные столбцы
        protected_columns = [
            self.config['data'].get('target_column'),
            self.config['data'].get('date_column'),
            self.config['data'].get('customer_id')
        ]
        protected_columns = [col for col in protected_columns if col is not None]
        
        cols_to_drop = [col for col in cols_to_drop if col not in protected_columns]
        
        if cols_to_drop:
            self.logger.info(f"Удаление столбцов с пропусками > {nan_threshold}: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # Статистика по оставшимся пропускам
        remaining_missing = df.isnull().sum()
        if remaining_missing.sum() > 0:
            self.logger.info(f"Столбцы с пропусками: {remaining_missing[remaining_missing > 0].to_dict()}")
        
        return df
    
    def handle_single_value_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление столбцов с одним уникальным значением"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Входные данные должны быть pandas.DataFrame")
            
        if df.empty:
            self.logger.warning("Пустой DataFrame передан в handle_single_value_columns")
            return df
            
        threshold = self.config.get('preprocessing', {}).get('single_value_threshold', 0.99)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            self.logger.warning(f"Неверный single_value_threshold: {threshold}, используем 0.99")
            threshold = 0.99
        
        cols_to_drop = []
        for col in df.columns:
            if col == self.config['data']['target_column']:
                continue
                
            # Проверяем долю наиболее частого значения
            value_counts = df[col].value_counts(normalize=True)
            if len(value_counts) == 1 or value_counts.iloc[0] > threshold:
                cols_to_drop.append(col)
        
        if cols_to_drop:
            self.logger.info(f"Удаление столбцов с низкой вариативностью: {len(cols_to_drop)} столбцов")
            df = df.drop(columns=cols_to_drop)
        
        return df
    
    def set_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Установка индекса"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Входные данные должны быть pandas.DataFrame")
            
        customer_id = self.config.get('data', {}).get('customer_id')
        if customer_id and customer_id in df.columns:
            try:
                df = df.set_index(customer_id)
                self.logger.info(f"Установлен индекс: {customer_id}")
            except Exception as e:
                self.logger.warning(f"Не удалось установить индекс {customer_id}: {e}")
        elif customer_id:
            self.logger.warning(f"Столбец для индекса '{customer_id}' не найден")
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Разбиение на train/validation/test"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Входные данные должны быть pandas.DataFrame")
            
        if df.empty:
            raise ValueError("Нельзя разбить пустой DataFrame")
            
        target_col = self.config['data']['target_column']
        if target_col not in df.columns:
            raise ValueError(f"Целевой столбец '{target_col}' отсутствует в данных")
            
        preprocessing_config = self.config.get('preprocessing', {})
        test_size = preprocessing_config.get('test_size', 0.2)
        val_size = preprocessing_config.get('val_size', 0.1) 
        stratify = preprocessing_config.get('stratify', True)
        
        # Валидация параметров
        if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
            raise ValueError(f"Неверный test_size: {test_size}")
        if not isinstance(val_size, (int, float)) or not 0 < val_size < 1:
            raise ValueError(f"Неверный val_size: {val_size}")
        if test_size + val_size >= 1:
            raise ValueError(f"Сумма test_size ({test_size}) и val_size ({val_size}) должна быть < 1")
        
        # Сначала отделяем test
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        stratify_param = y if stratify else None
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_size,
            stratify=stratify_param,
            random_state=self.config['random_seed']
        )
        
        # Затем разбиваем оставшиеся данные на train/val
        val_size_adjusted = val_size / (1 - test_size)
        stratify_param = y_temp if stratify else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=stratify_param,
            random_state=self.config['random_seed']
        )
        
        # Объединяем обратно X и y
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        self.logger.info(f"Разбиение данных:")
        self.logger.info(f"  Train: {train_df.shape[0]} строк ({train_df.shape[0]/len(df):.2%})")
        self.logger.info(f"  Validation: {val_df.shape[0]} строк ({val_df.shape[0]/len(df):.2%})")
        self.logger.info(f"  Test: {test_df.shape[0]} строк ({test_df.shape[0]/len(df):.2%})")
        
        # Проверка распределения целевой переменной
        self.logger.info("Распределение целевой переменной:")
        self.logger.info(f"  Train: {y_train.value_counts(normalize=True).to_dict()}")
        self.logger.info(f"  Validation: {y_val.value_counts(normalize=True).to_dict()}")
        self.logger.info(f"  Test: {y_test.value_counts(normalize=True).to_dict()}")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Сохранение обработанных данных"""
        # Валидация входных данных
        for name, df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{name}_df должен быть pandas.DataFrame")
            if df.empty:
                raise ValueError(f"{name}_df не может быть пустым")
                
        # Получаем путь из конфигурации
        output_dir = self.config.get('paths', {}).get('data_processed', 'data/processed')
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Не удалось создать директорию {output_dir}: {e}")
        
        try:
            train_df.to_csv(f"{output_dir}/train.csv", index=True)
            val_df.to_csv(f"{output_dir}/validation.csv", index=True)
            test_df.to_csv(f"{output_dir}/test.csv", index=True)
        except Exception as e:
            raise RuntimeError(f"Ошибка при сохранении данных: {e}")
        
        self.logger.info(f"Обработанные данные сохранены в {output_dir}/")
    
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Основной метод предобработки"""
        try:
            # Загрузка данных
            df = self.load_raw_data()
            
            # Применение фильтров
            df = self.apply_filters(df)
            
            # Проверка что остались данные после фильтрации
            if df.empty:
                raise ValueError("После применения фильтров не осталось данных")
            
            # Обработка пропущенных значений
            df = self.handle_missing_values(df)
            
            # Удаление столбцов с низкой вариативностью
            df = self.handle_single_value_columns(df)
            
            # Проверка что остались признаки
            if len(df.columns) <= 1:  # только target
                raise ValueError("После очистки не осталось признаков для обучения")
            
            # Установка индекса
            df = self.set_index(df)
            
            # Разбиение на train/val/test
            train_df, val_df, test_df = self.split_data(df)
            
            # Сохранение
            self.save_processed_data(train_df, val_df, test_df)
            
            self.logger.info("Предобработка успешно завершена")
            return train_df, val_df, test_df
            
        except Exception as e:
            self.logger.error(f"Ошибка в процессе предобработки: {e}")
            raise


def main() -> None:
    """Основная функция"""
    try:
        # Загрузка конфигурации
        config = load_config()
        
        # Создание препроцессора
        preprocessor = DataPreprocessor(config)
        
        # Обработка данных
        train_df, val_df, test_df = preprocessor.process()
        
        print("Предобработка данных завершена")
        print(f"Train: {train_df.shape}")
        print(f"Validation: {val_df.shape}")
        print(f"Test: {test_df.shape}")
        
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Проверьте наличие файла params.yaml и исходных данных")
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка валидации: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()