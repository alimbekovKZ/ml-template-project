#!/usr/bin/env python3
"""
Скрипт для обучения модели машинного обучения.
Поддерживает LightGBM, CatBoost и XGBoost.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import catboost as cb
except ImportError:
    cb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.append('src')
from utils.config import load_config
from utils.logging import setup_logger
from utils.mlflow_utils import MLflowTracker, setup_mlflow_autolog


class ModelTrainer:
    """Класс для обучения моделей ML"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger('ModelTrainer')
        self.model = None
        self.feature_importance = None
        
        # Инициализация MLflow tracker
        self.mlflow_tracker = MLflowTracker(config)
        
        # Настройка автоматического логирования
        if config.get('mlflow', {}).get('enabled', True):
            setup_mlflow_autolog(config['model']['type'])
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Загрузка обработанных данных"""
        train_path = "data/features/train_features.csv"
        val_path = "data/features/validation_features.csv"
        
        self.logger.info(f"Загрузка данных из {train_path} и {val_path}")
        
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        # Загрузка метаданных о признаках
        with open("data/features/feature_metadata.json", 'r') as f:
            feature_metadata = json.load(f)
            
        self.logger.info(f"Размер обучающей выборки: {train_df.shape}")
        self.logger.info(f"Размер валидационной выборки: {val_df.shape}")
        
        return train_df, val_df, feature_metadata
    
    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple:
        """Подготовка данных для обучения"""
        target_col = self.config['data']['target_column']
        
        # Разделение на признаки и целевую переменную
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        
        self.logger.info(f"Количество признаков: {X_train.shape[1]}")
        self.logger.info(f"Распределение целевой переменной в train: {y_train.value_counts(normalize=True).to_dict()}")
        
        return X_train, y_train, X_val, y_val
    
    def create_model(self) -> Any:
        """Создание модели в зависимости от типа"""
        model_type = self.config['model']['type'].lower()
        all_params = self.config['model']['hyperparameters']
        
        # Получаем параметры для конкретного типа модели
        params = all_params.get(model_type, {})
        
        if model_type == 'lightgbm':
            if lgb is None:
                raise ImportError("LightGBM не установлен или не может быть импортирован")
            model = lgb.LGBMClassifier(**params, random_state=self.config['random_seed'])
        elif model_type == 'catboost':
            if cb is None:
                raise ImportError("CatBoost не установлен")
            model = cb.CatBoostClassifier(**params, random_state=self.config['random_seed'], verbose=False)
        elif model_type == 'xgboost':
            if xgb is None:
                raise ImportError("XGBoost не установлен")
            model = xgb.XGBClassifier(**params, random_state=self.config['random_seed'])
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
            
        self.logger.info(f"Создана модель типа: {model_type}")
        return model
    
    def train_with_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Обучение с кросс-валидацией"""
        cv_folds = self.config['training']['cv_folds']
        
        if self.config['training']['stratified']:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config['random_seed'])
        else:
            cv = cv_folds
            
        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='roc_auc')
        
        cv_metrics = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        self.logger.info(f"CV AUC: {cv_metrics['cv_auc_mean']:.4f} (+/- {cv_metrics['cv_auc_std']*2:.4f})")
        
        return cv_metrics
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Основное обучение модели"""
        
        # Начинаем MLflow run
        self.mlflow_tracker.start_run(
            run_name=f"{self.config['experiment_name']}_{self.config['model']['type']}",
            tags={'stage': 'training'}
        )
        
        try:
            # Логируем параметры
            self.mlflow_tracker.log_params(self.config, "config")
            
            # Логируем информацию о данных
            data_info = {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'features_count': X_train.shape[1],
                'target_distribution_train': y_train.value_counts(normalize=True).to_dict(),
                'target_distribution_val': y_val.value_counts(normalize=True).to_dict()
            }
            self.mlflow_tracker.log_params(data_info, "data_info")
            
            self.model = self.create_model()
            
            # Кросс-валидация
            cv_metrics = self.train_with_cv(X_train, y_train)
            self.mlflow_tracker.log_metrics(cv_metrics, step=0)
            
            # Обучение на полной обучающей выборке
            if self.config['model']['type'].lower() == 'lightgbm' and lgb is not None:
                # Для LightGBM используем early stopping
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='auc',
                    callbacks=[lgb.early_stopping(self.config['training']['early_stopping'])]
                )
            else:
                self.model.fit(X_train, y_train)
            
            # Предсказания
            train_pred = self.model.predict_proba(X_train)[:, 1]
            val_pred = self.model.predict_proba(X_val)[:, 1]
            
            # Метрики
            train_metrics = self.calculate_metrics(y_train, train_pred)
            val_metrics = self.calculate_metrics(y_val, val_pred)
            
            # Логируем метрики в MLflow
            train_metrics_prefixed = {f"train_{k}": v for k, v in train_metrics.items()}
            val_metrics_prefixed = {f"val_{k}": v for k, v in val_metrics.items()}
            
            self.mlflow_tracker.log_metrics(train_metrics_prefixed, step=1)
            self.mlflow_tracker.log_metrics(val_metrics_prefixed, step=1)
            
            # Feature importance
            self.feature_importance = self.get_feature_importance(X_train.columns)
            
            # Логируем модель в MLflow
            input_example = X_val.head(5)
            self.mlflow_tracker.log_model(
                self.model, 
                self.config['model']['type'],
                artifact_path="model",
                input_example=input_example
            )
            
            # Логируем feature importance
            if not self.feature_importance.empty:
                self.mlflow_tracker.log_dataframe(
                    self.feature_importance, 
                    "feature_importance.csv", 
                    "features"
                )
            
            metrics = {
                'cross_validation': cv_metrics,
                'train': train_metrics,
                'validation': val_metrics
            }
            
            # Логируем общие метрики как JSON
            self.mlflow_tracker.log_dict_as_json(metrics, "all_metrics.json", "metrics")
            
            self.logger.info(f"Train AUC: {train_metrics['auc']:.4f}")
            self.logger.info(f"Validation AUC: {val_metrics['auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {e}")
            self.mlflow_tracker.end_run(status="FAILED")
            raise
    
    
    def calculate_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Расчет метрик качества"""
        threshold = self.config['evaluation']['thresholds']['default']
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
    
    def get_feature_importance(self, feature_names) -> pd.DataFrame:
        """Получение важности признаков"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
        else:
            self.logger.warning("Модель не поддерживает feature importance")
            return pd.DataFrame()
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def plot_feature_importance(self, save_path: str = "reports/figures/feature_importance.png"):
        """Визуализация важности признаков"""
        if self.feature_importance.empty:
            return
            
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(20)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Логируем в MLflow
        self.mlflow_tracker.log_figure(plt.gcf(), "feature_importance.png", "figures")
        
        plt.close()
        
        self.logger.info(f"График feature importance сохранен: {save_path}")
    
    def save_model(self, experiment_name: str):
        """Сохранение обученной модели"""
        model_dir = f"models/experiments/{experiment_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Сохранение модели
        model_path = f"{model_dir}/model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Сохранение feature importance
        if not self.feature_importance.empty:
            importance_path = f"{model_dir}/feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
        
        self.logger.info(f"Модель сохранена в {model_dir}")


def main():
    """Основная функция"""
    # Загрузка конфигурации
    config = load_config()
    
    # Создание тренера
    trainer = ModelTrainer(config)
    
    try:
        # Загрузка данных
        train_df, val_df, feature_metadata = trainer.load_data()
        
        # Подготовка данных
        X_train, y_train, X_val, y_val = trainer.prepare_data(train_df, val_df)
        
        # Обучение модели
        metrics = trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Сохранение метрик
        os.makedirs("reports/metrics", exist_ok=True)
        with open("reports/metrics/train_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Визуализация
        trainer.plot_feature_importance()
        
        # Сохранение модели
        experiment_name = config['experiment_name']
        trainer.save_model(experiment_name)
        
        # Логируем локальные артефакты в MLflow
        trainer.mlflow_tracker.log_artifact("reports/metrics/train_metrics.json", "metrics")
        trainer.mlflow_tracker.log_artifact("reports/figures/feature_importance.png", "figures")
        
        # Завершаем MLflow run
        trainer.mlflow_tracker.end_run()
        
        print(f"Обучение завершено. Эксперимент: {experiment_name}")
        print(f"Validation AUC: {metrics['validation']['auc']:.4f}")
        
    except Exception as e:
        # В случае ошибки завершаем run с ошибкой
        if hasattr(trainer, 'mlflow_tracker'):
            trainer.mlflow_tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()