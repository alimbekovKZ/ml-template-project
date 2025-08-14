#!/usr/bin/env python3
"""
Скрипт для создания признаков (feature engineering).
Включает кодирование категориальных переменных и масштабирование.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings


def convert_to_serializable(obj):
    """Преобразует объекты NumPy в JSON-сериализуемые типы"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')
from utils.config import load_config
from utils.logging import setup_logger


class FeatureEngineer:
    """Класс для создания признаков"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger('FeatureEngineer')
        self.categorical_encoders = {}
        self.numerical_scaler = None
        self.feature_selector = None
        self.feature_metadata = {}
        
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Загрузка предобработанных данных"""
        train_df = pd.read_csv("data/processed/train.csv", index_col=0)
        val_df = pd.read_csv("data/processed/validation.csv", index_col=0)
        test_df = pd.read_csv("data/processed/test.csv", index_col=0)
        
        self.logger.info(f"Загружены данные:")
        self.logger.info(f"  Train: {train_df.shape}")
        self.logger.info(f"  Validation: {val_df.shape}")
        self.logger.info(f"  Test: {test_df.shape}")
        
        return train_df, val_df, test_df
    
    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, list]:
        """Определение типов признаков"""
        target_col = self.config['data']['target_column']
        date_col = self.config['data']['date_column']
        
        # Исключаем целевую переменную и дату
        features = [col for col in df.columns if col not in [target_col, date_col]]
        
        categorical_features = []
        numerical_features = []
        
        for col in features:
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        feature_types = {
            'categorical': categorical_features,
            'numerical': numerical_features
        }
        
        self.logger.info(f"Типы признаков:")
        self.logger.info(f"  Категориальные: {len(categorical_features)}")
        self.logger.info(f"  Численные: {len(numerical_features)}")
        
        return feature_types
    
    def encode_categorical_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                                  test_df: pd.DataFrame, categorical_features: list) -> Tuple:
        """Кодирование категориальных признаков"""
        encoding_config = self.config['features']['categorical_encoding']
        method = encoding_config['method']
        
        if method == 'frequency':
            return self._frequency_encoding(train_df, val_df, test_df, categorical_features)
        elif method == 'target_encoding':
            return self._target_encoding(train_df, val_df, test_df, categorical_features)
        elif method == 'one_hot':
            return self._one_hot_encoding(train_df, val_df, test_df, categorical_features)
        else:
            raise ValueError(f"Неподдерживаемый метод кодирования: {method}")
    
    def _frequency_encoding(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                           test_df: pd.DataFrame, categorical_features: list) -> Tuple:
        """Частотное кодирование категориальных признаков"""
        cumsum_freq = self.config['features']['categorical_encoding']['cumsum_freq']
        
        for feature in categorical_features:
            # Вычисляем частоты на обучающей выборке
            value_counts = train_df[feature].value_counts()
            
            # Определяем категории для сохранения на основе cumsum_freq
            cumsum_normalized = value_counts.cumsum() / value_counts.sum()
            categories_to_keep = cumsum_normalized[cumsum_normalized <= cumsum_freq].index.tolist()
            
            # Создаем маппинг
            frequency_map = {}
            for cat in categories_to_keep:
                frequency_map[cat] = value_counts[cat]
            
            # Для редких категорий используем среднюю частоту редких категорий
            rare_categories = set(value_counts.index) - set(categories_to_keep)
            if rare_categories:
                rare_freq = value_counts[list(rare_categories)].mean()
                for cat in rare_categories:
                    frequency_map[cat] = rare_freq
            
            # Применяем кодирование
            for df in [train_df, val_df, test_df]:
                df[f"{feature}_encoded"] = df[feature].map(frequency_map).fillna(rare_freq if rare_categories else 0)
                df.drop(columns=[feature], inplace=True)
            
            self.categorical_encoders[feature] = frequency_map
        
        self.logger.info(f"Применено частотное кодирование для {len(categorical_features)} признаков")
        return train_df, val_df, test_df
    
    def _target_encoding(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, categorical_features: list) -> Tuple:
        """Target encoding категориальных признаков"""
        target_col = self.config['data']['target_column']
        
        for feature in categorical_features:
            # Вычисляем средние значения целевой переменной для каждой категории
            target_means = train_df.groupby(feature)[target_col].mean()
            global_mean = train_df[target_col].mean()
            
            # Применяем кодирование
            for df in [train_df, val_df, test_df]:
                df[f"{feature}_target_encoded"] = df[feature].map(target_means).fillna(global_mean)
                df.drop(columns=[feature], inplace=True)
            
            self.categorical_encoders[feature] = target_means.to_dict()
        
        self.logger.info(f"Применено target encoding для {len(categorical_features)} признаков")
        return train_df, val_df, test_df
    
    def _one_hot_encoding(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                         test_df: pd.DataFrame, categorical_features: list) -> Tuple:
        """One-hot encoding категориальных признаков"""
        for feature in categorical_features:
            # Получаем уникальные категории из обучающей выборки
            categories = train_df[feature].unique()
            
            # Применяем one-hot encoding
            for df in [train_df, val_df, test_df]:
                for category in categories:
                    df[f"{feature}_{category}"] = (df[feature] == category).astype(int)
                df.drop(columns=[feature], inplace=True)
            
            self.categorical_encoders[feature] = categories.tolist()
        
        self.logger.info(f"Применено one-hot encoding для {len(categorical_features)} признаков")
        return train_df, val_df, test_df
    
    def scale_numerical_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                               test_df: pd.DataFrame, numerical_features: list) -> Tuple:
        """Масштабирование численных признаков"""
        scaling_config = self.config['features']['numerical_scaling']
        method = scaling_config['method']
        
        if method == 'standard':
            self.numerical_scaler = StandardScaler()
        elif method == 'minmax':
            self.numerical_scaler = MinMaxScaler()
        elif method == 'robust':
            self.numerical_scaler = RobustScaler()
        else:
            self.logger.warning(f"Неподдерживаемый метод масштабирования: {method}. Пропускаем.")
            return train_df, val_df, test_df
        
        # Обучаем скейлер на обучающих данных
        self.numerical_scaler.fit(train_df[numerical_features])
        
        # Применяем масштабирование
        for df in [train_df, val_df, test_df]:
            df[numerical_features] = self.numerical_scaler.transform(df[numerical_features])
        
        self.logger.info(f"Применено масштабирование '{method}' для {len(numerical_features)} признаков")
        return train_df, val_df, test_df
    
    def apply_feature_selection(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                              test_df: pd.DataFrame) -> Tuple:
        """Отбор признаков"""
        selection_config = self.config['features']['feature_selection']
        
        if not selection_config['enabled']:
            return train_df, val_df, test_df
        
        target_col = self.config['data']['target_column']
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        method = selection_config['method']
        k = selection_config['top_k']
        
        if method == 'importance':
            # Будет реализовано после обучения модели
            self.logger.info("Feature selection на основе важности будет применен после обучения модели")
            return train_df, val_df, test_df
        elif method == 'correlation':
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'mutual_info':
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            self.logger.warning(f"Неподдерживаемый метод отбора признаков: {method}")
            return train_df, val_df, test_df
        
        # Обучаем селектор признаков
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        selected_features = X_train.columns[self.feature_selector.get_support()].tolist()
        
        # Применяем отбор ко всем наборам данных
        for df in [train_df, val_df, test_df]:
            features_to_keep = [target_col] + selected_features
            df = df[features_to_keep]
        
        self.logger.info(f"Отобрано {len(selected_features)} признаков из {X_train.shape[1]}")
        return train_df, val_df, test_df
    
    def save_feature_metadata(self):
        """Сохранение метаданных о признаках"""
        self.feature_metadata = {
            'categorical_encoders': self.categorical_encoders,
            'numerical_scaler': self.numerical_scaler,
            'feature_selector': self.feature_selector,
            'config': self.config['features']
        }
        
        # Сохраняем в JSON (для параметров) и pickle (для объектов sklearn)
        os.makedirs("data/features", exist_ok=True)
        
        # JSON метаданные
        json_metadata = {
            'categorical_encoders': convert_to_serializable(self.categorical_encoders),
            'config': convert_to_serializable(self.config['features'])
        }
        
        with open("data/features/feature_metadata.json", 'w') as f:
            json.dump(json_metadata, f, indent=2)
        
        # Pickle объекты sklearn
        sklearn_objects = {
            'numerical_scaler': self.numerical_scaler,
            'feature_selector': self.feature_selector
        }
        
        with open("data/features/sklearn_objects.pkl", 'wb') as f:
            pickle.dump(sklearn_objects, f)
        
        self.logger.info("Метаданные признаков сохранены")
    
    def save_processed_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Сохранение обработанных признаков"""
        output_dir = "data/features"
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(f"{output_dir}/train_features.csv", index=True)
        val_df.to_csv(f"{output_dir}/validation_features.csv", index=True)
        test_df.to_csv(f"{output_dir}/test_features.csv", index=True)
        
        self.logger.info(f"Признаки сохранены в {output_dir}/")
    
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Основной метод создания признаков"""
        # Загрузка данных
        train_df, val_df, test_df = self.load_processed_data()
        
        # Определение типов признаков
        feature_types = self.identify_feature_types(train_df)
        
        # Кодирование категориальных признаков
        if feature_types['categorical']:
            train_df, val_df, test_df = self.encode_categorical_features(
                train_df, val_df, test_df, feature_types['categorical']
            )
        
        # Обновляем список численных признаков после кодирования
        target_col = self.config['data']['target_column']
        date_col = self.config['data']['date_column']
        # Исключаем целевую переменную, дату и категориальные признаки
        numerical_features = [col for col in train_df.columns 
                            if col not in [target_col, date_col] and 
                            train_df[col].dtype in ['int64', 'float64']]
        
        # Масштабирование численных признаков
        if numerical_features:
            train_df, val_df, test_df = self.scale_numerical_features(
                train_df, val_df, test_df, numerical_features
            )
        
        # Отбор признаков
        train_df, val_df, test_df = self.apply_feature_selection(train_df, val_df, test_df)
        
        # Сохранение метаданных
        self.save_feature_metadata()
        
        # Сохранение обработанных данных
        self.save_processed_features(train_df, val_df, test_df)
        
        return train_df, val_df, test_df


def main():
    """Основная функция"""
    # Загрузка конфигурации
    config = load_config()
    
    # Создание feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Обработка признаков
    train_df, val_df, test_df = feature_engineer.process()
    
    print("Feature engineering завершен")
    print(f"Train features: {train_df.shape}")
    print(f"Validation features: {val_df.shape}")
    print(f"Test features: {test_df.shape}")


if __name__ == "__main__":
    main()