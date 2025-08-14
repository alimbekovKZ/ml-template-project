#!/usr/bin/env python3
"""
Скрипт для загрузки и первичной подготовки данных.
Предполагает загрузку данных из внешнего источника в raw директорию.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')
from utils.config import load_config
from utils.logging import setup_logger


class DataLoader:
    """Класс для загрузки данных"""
    
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError("config должен быть словарем")
        
        self.config = config
        self.logger = setup_logger('DataLoader')
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        if 'data' not in self.config:
            raise ValueError("Отсутствует секция 'data' в конфигурации")
        
        required_fields = ['target_column', 'date_column', 'customer_id']
        for field in required_fields:
            if field not in self.config['data']:
                raise ValueError(f"Отсутствует поле '{field}' в секции data")
        
    def _get_source_path(self) -> str:
        """Получение пути к источнику данных"""
        return self.config['data'].get('source_path', 'data/external/raw_data.csv')
        
    def _get_output_path(self, filename: str = "dataset.csv") -> str:
        """Получение выходного пути"""
        base_path = self.config.get('paths', {}).get('data_raw', 'data/raw')
        return f"{base_path}/{filename}"
    
    def load_external_data(self) -> pd.DataFrame:
        """
        Загрузка данных из внешнего источника.
        В реальном проекте здесь может быть:
        - Подключение к базе данных
        - API запросы
        - Загрузка из облачного хранилища
        """
        source_path = self._get_source_path()
        
        if os.path.exists(source_path):
            try:
                self.logger.info(f"Загрузка данных из {source_path}")
                
                # Проверяем что файл не пустой
                if os.path.getsize(source_path) == 0:
                    raise ValueError(f"Файл {source_path} пустой")
                
                df = pd.read_csv(source_path)
                
                # Базовая валидация
                if df.empty:
                    raise ValueError(f"Файл {source_path} не содержит данных")
                
                self.logger.info(f"Загружено {len(df)} строк, {len(df.columns)} столбцов")
                return df
                
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                self.logger.error(f"Ошибка парсинга файла {source_path}: {e}")
                raise ValueError(f"Неверный формат файла: {e}")
            except Exception as e:
                self.logger.error(f"Ошибка чтения файла {source_path}: {e}")
                raise
        else:
            # Создаем демо данные если файл не существует
            self.logger.warning(f"Файл {source_path} не найден. Создаем демо данные.")
            return self.create_demo_data()
    
    def create_demo_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Создание демонстрационных данных для тестирования пайплайна"""
        np.random.seed(self.config['random_seed'])
        
        # Генерируем синтетические данные
        data = {
            'the_date': pd.date_range('2023-01-01', periods=n_samples, freq='D')[:n_samples],
            'abs_customer': [f'customer_{i:05d}' for i in range(n_samples)],
            'age': np.random.randint(18, 80, n_samples),
            'disbursement': np.random.lognormal(mean=7, sigma=1, size=n_samples),
            'income': np.random.lognormal(mean=10, sigma=0.5, size=n_samples),
            'credit_score': np.random.normal(650, 100, n_samples),
            'employment_length': np.random.randint(0, 30, n_samples),
            'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples),
            'purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'small_business'], n_samples),
            'education': np.random.choice(['High School', 'College', 'Graduate'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        }
        
        # Создаем целевую переменную с некоторой логикой
        base_prob = 0.1  # базовая вероятность дефолта
        
        # Факторы влияющие на вероятность
        age_factor = np.where(data['age'] < 25, 1.5, 
                             np.where(data['age'] > 65, 1.2, 1.0))
        score_factor = np.where(data['credit_score'] < 600, 2.0,
                               np.where(data['credit_score'] > 750, 0.5, 1.0))
        income_factor = np.where(data['income'] < 30000, 1.8,
                                np.where(data['income'] > 100000, 0.6, 1.0))
        
        # Финальная вероятность
        prob = base_prob * age_factor * score_factor * income_factor
        prob = np.clip(prob, 0.01, 0.95)
        
        data['target'] = np.random.binomial(1, prob, n_samples)
        
        df = pd.DataFrame(data)
        
        # Добавляем некоторые пропущенные значения
        missing_cols = ['income', 'employment_length', 'credit_score']
        for col in missing_cols:
            missing_mask = np.random.random(len(df)) < 0.05  # 5% пропусков
            df.loc[missing_mask, col] = np.nan
        
        self.logger.info(f"Создано {len(df)} строк демонстрационных данных")
        self.logger.info(f"Распределение целевой переменной: {df['target'].value_counts(normalize=True).to_dict()}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Базовая валидация загруженных данных"""
        required_columns = [
            self.config['data']['target_column'],
            self.config['data']['date_column'],
            self.config['data']['customer_id']
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            self.logger.error(f"Отсутствуют обязательные столбцы: {missing_cols}")
            return False
        
        # Проверяем что целевая переменная бинарная
        target_col = self.config['data']['target_column']
        unique_targets = df[target_col].unique()
        if len(unique_targets) != 2 or not set(unique_targets).issubset({0, 1}):
            self.logger.error(f"Целевая переменная должна быть бинарной (0/1), найдено: {unique_targets}")
            return False
        
        # Проверяем формат даты
        date_col = self.config['data']['date_column']
        try:
            pd.to_datetime(df[date_col])
        except Exception as e:
            self.logger.error(f"Ошибка в формате даты в столбце {date_col}: {e}")
            return False
        
        # Проверяем уникальность customer_id
        customer_col = self.config['data']['customer_id']
        if df[customer_col].duplicated().any():
            self.logger.warning(f"Найдены дублированные значения в {customer_col}")
        
        self.logger.info("Валидация данных прошла успешно")
        return True
    
    def save_raw_data(self, df: pd.DataFrame) -> None:
        """Сохранение данных в raw директорию"""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Входные данные должны быть pandas.DataFrame")
            
        if df.empty:
            raise ValueError("Нельзя сохранить пустой DataFrame")
        
        output_path = self._get_output_path()
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Не удалось создать директорию: {e}")
        
        try:
            df.to_csv(output_path, index=False)
            self.logger.info(f"Данные сохранены: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Ошибка сохранения данных: {e}")
        
        # Сохраняем метаданные
        try:
            target_col = self.config['data']['target_column']
            metadata = {
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'target_distribution': df[target_col].value_counts().to_dict() if target_col in df.columns else {},
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
            
            import json
            metadata_path = os.path.join(os.path.dirname(output_path), "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"Метаданные сохранены: {metadata_path}")
        except Exception as e:
            self.logger.warning(f"Ошибка сохранения метаданных: {e}")
        
    
    def process(self) -> pd.DataFrame:
        """Основной метод обработки"""
        # Загрузка данных
        df = self.load_external_data()
        
        # Валидация
        if not self.validate_data(df):
            raise ValueError("Данные не прошли валидацию")
        
        # Сохранение
        self.save_raw_data(df)
        
        return df


def main() -> None:
    """Основная функция"""
    try:
        # Загрузка конфигурации
        config = load_config()
        
        # Создание загрузчика данных
        loader = DataLoader(config)
        
        # Загрузка и обработка данных
        df = loader.process()
        
        print("Загрузка данных завершена")
        print(f"Размер данных: {df.shape}")
        
        output_path = loader._get_output_path()
        print(f"Данные сохранены в: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Ошибка: {e}")
        print("Проверьте наличие файла params.yaml")
        sys.exit(1)
    except ValueError as e:
        print(f"Ошибка валидации: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()