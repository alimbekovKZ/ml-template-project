#!/usr/bin/env python3
"""
Скрипт для обучения модели машинного обучения с поддержкой Hyperopt.
Поддерживает CatBoost и XGBoost.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Импорты для моделей ML
cb = None
xgb = None
hyperopt_available = False

try:
    import catboost as cb
    print("CatBoost успешно импортирован")
except ImportError:
    print("CatBoost не установлен")

try:
    import xgboost as xgb
    print("XGBoost успешно импортирован")
except ImportError:
    print("XGBoost не установлен")

try:
    from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
    hyperopt_available = True
    print("Hyperopt успешно импортирован")
except ImportError:
    print("Hyperopt не установлен - оптимизация гиперпараметров недоступна")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import sys
sys.path.append('src')
from utils.config import load_config
from utils.logging import setup_logger
from utils.mlflow_utils import MLflowTracker, setup_mlflow_autolog


class ModelTrainer:
    """Класс для обучения моделей ML с поддержкой Hyperopt"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger('ModelTrainer')
        self.model = None
        self.feature_importance = None
        self.best_params = None
        self.best_score = None
        
        # Данные для оптимизации
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
        # Инициализация MLflow tracker
        self.mlflow_tracker = MLflowTracker(config)
        
        # Настройка автоматического логирования
        if config.get('mlflow', {}).get('enabled', True):
            setup_mlflow_autolog(config['model']['type'])
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Загрузка обработанных данных"""
        train_path = "data/features/train_features.csv"
        val_path = "data/features/validation_features.csv"
        
        self.logger.info(f"Загрузка данных из {train_path} и {val_path}")
        
        train_df = pd.read_csv(train_path, index_col=0)
        val_df = pd.read_csv(val_path, index_col=0)
        
        # Загрузка метаданных о признаках
        try:
            with open("data/features/feature_metadata.json", 'r') as f:
                feature_metadata = json.load(f)
        except FileNotFoundError:
            self.logger.warning("Метаданные признаков не найдены")
            feature_metadata = {}
            
        self.logger.info(f"Размер обучающей выборки: {train_df.shape}")
        self.logger.info(f"Размер валидационной выборки: {val_df.shape}")
        
        return train_df, val_df, feature_metadata
    
    def prepare_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple:
        """Подготовка данных для обучения"""
        target_col = self.config['data']['target_column']
        date_col = self.config['data']['date_column']
        
        # Колонки для исключения
        columns_to_exclude = [target_col]
        if date_col in train_df.columns:
            columns_to_exclude.append(date_col)
        
        # Исключаем ID колонки
        id_columns = [col for col in train_df.columns if 'customer' in col.lower() or 'id' in col.lower()]
        columns_to_exclude.extend(id_columns)
        
        # Разделение на признаки и целевую переменную
        X_train = train_df.drop(columns=columns_to_exclude)
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=columns_to_exclude)
        y_val = val_df[target_col]
        
        self.logger.info(f"Количество признаков: {X_train.shape[1]}")
        
        # Сохраняем для использования в оптимизации
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        return X_train, y_train, X_val, y_val
    
    def create_hyperopt_space(self) -> dict:
        """Создание пространства поиска для Hyperopt из конфигурации"""
        hyperopt_config = self.config.get('hyperopt', {})
        search_space = hyperopt_config.get('search_space', {})
        model_type = self.config['model']['type'].lower()
        
        space = {}
        
        if model_type == 'catboost':
            # Пространство для CatBoost
            if 'learning_rate' in search_space:
                lr_range = search_space['learning_rate']
                space['learning_rate'] = hp.uniform('learning_rate', lr_range[0], lr_range[1])
            
            if 'depth' in search_space:
                depth_range = search_space.get('depth', [4, 10])
                space['depth'] = hp.randint('depth', depth_range[0], depth_range[1] + 1)
            
            if 'l2_leaf_reg' in search_space:
                l2_range = search_space.get('l2_leaf_reg', [1, 10])
                space['l2_leaf_reg'] = hp.uniform('l2_leaf_reg', l2_range[0], l2_range[1])
                
            # Добавляем дефолтные параметры если пространство пустое
            if not space:
                space = {
                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
                    'depth': hp.randint('depth', 4, 11),
                    'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10)
                }
                
        elif model_type == 'xgboost':
            # Пространство для XGBoost
            if 'learning_rate' in search_space:
                lr_range = search_space['learning_rate']
                space['learning_rate'] = hp.uniform('learning_rate', lr_range[0], lr_range[1])
            
            if 'max_depth' in search_space:
                depth_range = search_space.get('max_depth', [3, 10])
                space['max_depth'] = hp.randint('max_depth', depth_range[0], depth_range[1] + 1)
            
            if 'subsample' in search_space:
                subsample_range = search_space.get('subsample', [0.5, 1.0])
                space['subsample'] = hp.uniform('subsample', subsample_range[0], subsample_range[1])
            
            if 'colsample_bytree' in search_space:
                colsample_range = search_space.get('colsample_bytree', [0.5, 1.0])
                space['colsample_bytree'] = hp.uniform('colsample_bytree', colsample_range[0], colsample_range[1])
            
            # Добавляем дефолтные параметры если пространство пустое
            if not space:
                space = {
                    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
                    'max_depth': hp.randint('max_depth', 3, 11),
                    'subsample': hp.uniform('subsample', 0.5, 1.0),
                    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0)
                }
        
        # Для других поисковых параметров из конфигурации
        for param, bounds in search_space.items():
            if param not in space:
                if isinstance(bounds, list) and len(bounds) == 2:
                    if all(isinstance(x, int) for x in bounds):
                        space[param] = hp.randint(param, bounds[0], bounds[1] + 1)
                    else:
                        space[param] = hp.uniform(param, bounds[0], bounds[1])
        
        return space
    
    def objective_function(self, params: dict) -> dict:
        """Целевая функция для оптимизации Hyperopt"""
        model_type = self.config['model']['type'].lower()
        
        # Преобразуем параметры (hp.randint возвращает float)
        if 'depth' in params:
            params['depth'] = int(params['depth'])
        if 'max_depth' in params:
            params['max_depth'] = int(params['max_depth'])
        
        # Создаем модель с текущими параметрами
        if model_type == 'catboost' and cb is not None:
            # Берем базовые параметры и обновляем их
            base_params = self.config['model']['hyperparameters'].get('catboost', {}).copy()
            base_params.update(params)
            base_params['random_state'] = self.config['random_seed']
            base_params['verbose'] = False
            
            model = cb.CatBoostClassifier(**base_params)
            
            # Обучение с early stopping
            model.fit(
                self.X_train, self.y_train,
                eval_set=(self.X_val, self.y_val),
                early_stopping_rounds=50,
                verbose=False
            )
            
        elif model_type == 'xgboost' and xgb is not None:
            base_params = self.config['model']['hyperparameters'].get('xgboost', {}).copy()
            base_params.update(params)
            base_params['random_state'] = self.config['random_seed']
            base_params['verbosity'] = 0
            
            model = xgb.XGBClassifier(**base_params)
            model.fit(self.X_train, self.y_train)
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        # Оценка на валидации
        val_pred_proba = model.predict_proba(self.X_val)[:, 1]
        val_auc = roc_auc_score(self.y_val, val_pred_proba)
        
        # Логируем в MLflow каждую итерацию
        if self.config.get('mlflow', {}).get('enabled', True):
            import mlflow
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric('val_auc', val_auc)
        
        # Hyperopt минимизирует, поэтому возвращаем отрицательный AUC
        return {'loss': -val_auc, 'status': STATUS_OK}
    
    def optimize_hyperparameters(self) -> Tuple[dict, float]:
        """Оптимизация гиперпараметров с помощью Hyperopt"""
        if not hyperopt_available:
            self.logger.warning("Hyperopt не установлен, используем параметры по умолчанию")
            return None, None
        
        hyperopt_config = self.config.get('hyperopt', {})
        
        if not hyperopt_config.get('enabled', False):
            self.logger.info("Оптимизация гиперпараметров отключена")
            return None, None
        
        self.logger.info("Начинаем оптимизацию гиперпараметров...")
        
        # Создаем пространство поиска
        space = self.create_hyperopt_space()
        
        if not space:
            self.logger.warning("Пространство поиска пустое, пропускаем оптимизацию")
            return None, None
        
        # Параметры оптимизации
        n_trials = hyperopt_config.get('n_trials', 50)
        
        # Запускаем оптимизацию
        trials = Trials()
        
        # Начинаем родительский MLflow run для оптимизации
        self.mlflow_tracker.start_run(
            run_name=f"hyperopt_{self.config['experiment_name']}",
            tags={'stage': 'hyperparameter_optimization', 'optimizer': 'hyperopt'}
        )
        
        try:
            best = fmin(
                fn=self.objective_function,
                space=space,
                algo=tpe.suggest,
                max_evals=n_trials,
                trials=trials,
                verbose=True
            )
            
            # Получаем лучшие параметры
            best_params = space_eval(space, best)
            
            # Лучший скор (помним что мы минимизировали отрицательный AUC)
            best_score = -min(trials.losses())
            
            self.logger.info(f"Лучший AUC: {best_score:.4f}")
            self.logger.info(f"Лучшие параметры: {best_params}")
            
            # Логируем лучшие результаты
            self.mlflow_tracker.log_params(best_params, prefix="best")
            self.mlflow_tracker.log_metrics({'best_val_auc': best_score})
            
            # Сохраняем историю оптимизации
            optimization_history = {
                'best_params': best_params,
                'best_score': best_score,
                'n_trials': n_trials,
                'all_results': [
                    {'params': t['misc']['vals'], 'loss': t['result']['loss']} 
                    for t in trials.trials
                ]
            }
            
            self.mlflow_tracker.log_dict_as_json(
                optimization_history, 
                "hyperopt_history.json", 
                "optimization"
            )
            
            return best_params, best_score
            
        finally:
            self.mlflow_tracker.end_run()
    
    def create_model(self, params: Optional[dict] = None) -> Any:
        """Создание модели с заданными параметрами"""
        model_type = self.config['model']['type'].lower()
        
        # Получаем базовые параметры
        hyperparams = self.config['model']['hyperparameters']
        
        if model_type == 'catboost':
            if cb is None:
                raise ImportError("CatBoost не установлен")
            
            base_params = hyperparams.get('catboost', hyperparams).copy()
            
            # Обновляем параметрами из оптимизации если есть
            if params:
                base_params.update(params)
            
            base_params['random_state'] = self.config['random_seed']
            model = cb.CatBoostClassifier(**base_params)
            
        elif model_type == 'xgboost':
            if xgb is None:
                raise ImportError("XGBoost не установлен")
            
            base_params = hyperparams.get('xgboost', hyperparams).copy()
            
            if params:
                base_params.update(params)
            
            base_params['random_state'] = self.config['random_seed']
            model = xgb.XGBClassifier(**base_params)
            
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        self.logger.info(f"Создана модель типа: {model_type}")
        if params:
            self.logger.info(f"С оптимизированными параметрами: {params}")
        
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
        
        # Сначала оптимизируем гиперпараметры если включено
        best_params, best_score = self.optimize_hyperparameters()
        
        # Сохраняем лучшие параметры
        self.best_params = best_params
        self.best_score = best_score
        
        # Начинаем MLflow run для финального обучения
        run_name = f"{self.config['experiment_name']}_{self.config['model']['type']}"
        if best_params:
            run_name += "_optimized"
            
        self.mlflow_tracker.start_run(
            run_name=run_name,
            tags={
                'stage': 'training',
                'optimized': str(best_params is not None)
            }
        )
        
        try:
            # Логируем параметры
            self.mlflow_tracker.log_params(self.config, "config")
            
            if best_params:
                self.mlflow_tracker.log_params(best_params, "optimized_params")
                self.mlflow_tracker.log_metrics({'hyperopt_best_auc': best_score})
            
            # Логируем информацию о данных
            data_info = {
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'features_count': X_train.shape[1],
                'target_distribution_train': y_train.value_counts(normalize=True).to_dict(),
                'target_distribution_val': y_val.value_counts(normalize=True).to_dict()
            }
            self.mlflow_tracker.log_params(data_info, "data_info")
            
            # Создаем модель с лучшими параметрами
            self.model = self.create_model(best_params)
            
            # Кросс-валидация
            cv_metrics = self.train_with_cv(X_train, y_train)
            self.mlflow_tracker.log_metrics(cv_metrics, step=0)
            
            # Обучение на полной обучающей выборке
            model_type = self.config['model']['type'].lower()
            
            if model_type == 'catboost' and cb is not None:
                self.model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=self.config['training']['early_stopping'],
                    verbose=False
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
                'validation': val_metrics,
                'hyperopt': {
                    'enabled': best_params is not None,
                    'best_params': best_params,
                    'best_score': best_score
                }
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
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
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
        
        # Сохранение лучших параметров если есть
        if self.best_params:
            params_path = f"{model_dir}/best_params.json"
            with open(params_path, 'w') as f:
                json.dump({
                    'best_params': self.best_params,
                    'best_score': self.best_score
                }, f, indent=2)
        
        self.logger.info(f"Модель сохранена в {model_dir}")


def main():
    """Основная функция"""
    # Загрузка конфигурации
    config = load_config()
    
    print(f"Тип модели для обучения: {config['model']['type']}")
    
    hyperopt_enabled = config.get('hyperopt', {}).get('enabled', False)
    if hyperopt_enabled:
        print(f"Оптимизация гиперпараметров: включена")
        print(f"Количество итераций: {config.get('hyperopt', {}).get('n_trials', 50)}")
    else:
        print("Оптимизация гиперпараметров: отключена")
    
    # Создание тренера
    trainer = ModelTrainer(config)
    
    try:
        # Загрузка данных
        train_df, val_df, feature_metadata = trainer.load_data()
        
        # Подготовка данных
        X_train, y_train, X_val, y_val = trainer.prepare_data(train_df, val_df)
        
        # Обучение модели (с оптимизацией если включена)
        metrics = trainer.train_model(X_train, y_train, X_val, y_val)
        
        # Сохранение метрик
        os.makedirs("reports/metrics", exist_ok=True)
        with open("reports/metrics/train_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
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
        
        print(f"\nОбучение завершено. Эксперимент: {experiment_name}")
        print(f"Validation AUC: {metrics['validation']['auc']:.4f}")
        
        if metrics['hyperopt']['enabled']:
            print(f"\nОптимизация гиперпараметров:")
            print(f"Лучший AUC: {metrics['hyperopt']['best_score']:.4f}")
            print(f"Лучшие параметры: {metrics['hyperopt']['best_params']}")
        
    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        # В случае ошибки завершаем run с ошибкой
        if hasattr(trainer, 'mlflow_tracker'):
            trainer.mlflow_tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()