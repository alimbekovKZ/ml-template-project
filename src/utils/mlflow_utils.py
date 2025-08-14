"""
Утилиты для работы с MLflow.
"""

import os
import json
import pickle
import tempfile
from typing import Dict, Any, Optional
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.catboost
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .logging import setup_logger


class MLflowTracker:
    """Класс для трекинга экспериментов в MLflow"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger('MLflowTracker')
        self.client = MlflowClient()
        
        # Настройка MLflow
        if config.get('mlflow', {}).get('enabled', True):
            tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db')
            mlflow.set_tracking_uri(tracking_uri)
            
            experiment_name = config.get('mlflow', {}).get('experiment_name', 'default')
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(experiment_name)
                    self.logger.info(f"Создан новый эксперимент: {experiment_name}")
                else:
                    experiment_id = experiment.experiment_id
                    
                mlflow.set_experiment(experiment_name)
                self.experiment_id = experiment_id
                
            except Exception as e:
                self.logger.warning(f"Ошибка при настройке MLflow: {e}")
                self.experiment_id = None
        else:
            self.experiment_id = None
            self.logger.info("MLflow отключен в конфигурации")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """Начало нового run"""
        if self.experiment_id is None:
            return None
            
        run_name = run_name or self.config.get('experiment_name', 'default_run')
        
        # Добавляем стандартные теги
        default_tags = {
            'model_type': self.config.get('model', {}).get('type', 'unknown'),
            'data_source': self.config.get('data', {}).get('source_path', 'unknown'),
            'experiment_name': self.config.get('experiment_name', 'default')
        }
        
        if tags:
            default_tags.update(tags)
        
        mlflow.start_run(run_name=run_name, tags=default_tags)
        self.logger.info(f"Начат MLflow run: {run_name}")
        
    def log_params(self, params: Dict[str, Any], prefix: str = ""):
        """Логирование параметров"""
        if mlflow.active_run() is None:
            return
            
        flat_params = self._flatten_dict(params, prefix)
        for key, value in flat_params.items():
            try:
                # MLflow не поддерживает сложные объекты как параметры
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                else:
                    mlflow.log_param(key, str(value))
            except Exception as e:
                self.logger.warning(f"Не удалось залогировать параметр {key}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Логирование метрик"""
        if mlflow.active_run() is None:
            return
            
        for key, value in metrics.items():
            try:
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    mlflow.log_metric(key, value, step=step)
            except Exception as e:
                self.logger.warning(f"Не удалось залогировать метрику {key}: {e}")
    
    def log_model(self, model, model_type: str, artifact_path: str = "model", 
                  signature=None, input_example=None):
        """Логирование модели"""
        if mlflow.active_run() is None:
            return
            
        try:
            if model_type.lower() == 'lightgbm':
                mlflow.lightgbm.log_model(
                    model, artifact_path, 
                    signature=signature, 
                    input_example=input_example
                )
            elif model_type.lower() == 'catboost':
                mlflow.catboost.log_model(
                    model, artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            elif model_type.lower() == 'xgboost':
                mlflow.xgboost.log_model(
                    model, artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            else:
                mlflow.sklearn.log_model(
                    model, artifact_path,
                    signature=signature,
                    input_example=input_example
                )
            
            self.logger.info(f"Модель {model_type} залогирована в MLflow")
            
        except Exception as e:
            self.logger.error(f"Ошибка при логировании модели: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Логирование артефакта"""
        if mlflow.active_run() is None:
            return
            
        try:
            if os.path.isfile(local_path):
                mlflow.log_artifact(local_path, artifact_path)
            elif os.path.isdir(local_path):
                mlflow.log_artifacts(local_path, artifact_path)
            
            self.logger.info(f"Артефакт залогирован: {local_path}")
            
        except Exception as e:
            self.logger.warning(f"Ошибка при логировании артефакта {local_path}: {e}")
    
    def log_figure(self, fig, filename: str, artifact_path: str = "figures"):
        """Логирование matplotlib фигуры"""
        if mlflow.active_run() is None:
            return
            
        try:
            with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
                fig.savefig(tmp.name, dpi=300, bbox_inches='tight')
                mlflow.log_artifact(tmp.name, artifact_path)
                os.unlink(tmp.name)
                
            self.logger.info(f"График залогирован: {filename}")
            
        except Exception as e:
            self.logger.warning(f"Ошибка при логировании графика {filename}: {e}")
    
    def log_dataframe(self, df: pd.DataFrame, filename: str, artifact_path: str = "data"):
        """Логирование DataFrame"""
        if mlflow.active_run() is None:
            return
            
        try:
            with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
                if filename.endswith('.parquet'):
                    df.to_parquet(tmp.name)
                else:
                    df.to_csv(tmp.name, index=False)
                    
                mlflow.log_artifact(tmp.name, artifact_path)
                os.unlink(tmp.name)
                
            self.logger.info(f"DataFrame залогирован: {filename}")
            
        except Exception as e:
            self.logger.warning(f"Ошибка при логировании DataFrame {filename}: {e}")
    
    def log_dict_as_json(self, data: Dict[str, Any], filename: str, 
                        artifact_path: str = "configs"):
        """Логирование словаря как JSON"""
        if mlflow.active_run() is None:
            return
            
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix=f"_{filename}", delete=False) as tmp:
                json.dump(data, tmp, indent=2, default=str)
                tmp.flush()
                mlflow.log_artifact(tmp.name, artifact_path)
                os.unlink(tmp.name)
                
            self.logger.info(f"JSON залогирован: {filename}")
            
        except Exception as e:
            self.logger.warning(f"Ошибка при логировании JSON {filename}: {e}")
    
    def end_run(self, status: str = "FINISHED"):
        """Завершение run"""
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
            self.logger.info("MLflow run завершен")
    
    def get_best_run(self, metric_name: str, ascending: bool = False):
        """Получение лучшего run по метрике"""
        if self.experiment_id is None:
            return None
            
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
                max_results=1
            )
            
            if runs:
                return runs[0]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка при поиске лучшего run: {e}")
            return None
    
    def compare_runs(self, run_ids: list, metrics: list = None):
        """Сравнение run'ов"""
        if self.experiment_id is None:
            return None
            
        try:
            runs_data = []
            for run_id in run_ids:
                run = self.client.get_run(run_id)
                run_data = {
                    'run_id': run_id,
                    'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'status': run.info.status,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time
                }
                
                # Добавляем метрики
                if metrics:
                    for metric in metrics:
                        run_data[f'metric_{metric}'] = run.data.metrics.get(metric, None)
                else:
                    for metric_name, value in run.data.metrics.items():
                        run_data[f'metric_{metric_name}'] = value
                
                runs_data.append(run_data)
            
            return pd.DataFrame(runs_data)
            
        except Exception as e:
            self.logger.error(f"Ошибка при сравнении run'ов: {e}")
            return None
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Рекурсивное разворачивание вложенного словаря"""
        items = []
        
        for key, value in d.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key).items())
            else:
                items.append((new_key, value))
        
        return dict(items)


def setup_mlflow_autolog(model_type: str, disable_existing: bool = True):
    """Настройка автоматического логирования для разных типов моделей"""
    if disable_existing:
        mlflow.sklearn.autolog(disable=True)
        mlflow.lightgbm.autolog(disable=True)
        if hasattr(mlflow, 'catboost'):
            mlflow.catboost.autolog(disable=True)
        if hasattr(mlflow, 'xgboost'):
            mlflow.xgboost.autolog(disable=True)
    
    if model_type.lower() == 'lightgbm':
        mlflow.lightgbm.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True
        )
    elif model_type.lower() == 'catboost':
        if hasattr(mlflow, 'catboost'):
            mlflow.catboost.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True
            )
    elif model_type.lower() == 'xgboost':
        if hasattr(mlflow, 'xgboost'):
            mlflow.xgboost.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True
            )
    else:
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            log_models=True
        )