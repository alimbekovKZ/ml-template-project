#!/usr/bin/env python3
"""
Скрипт для оценки обученной модели с интеграцией MLflow.
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, log_loss, confusion_matrix, roc_curve, 
    precision_recall_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('src')
from utils.config import load_config
from utils.logging import setup_logger
from utils.mlflow_utils import MLflowTracker


class ModelEvaluator:
    """Класс для оценки моделей с интеграцией MLflow"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger('ModelEvaluator')
        
        # Инициализация MLflow tracker
        self.mlflow_tracker = MLflowTracker(config)
        
    def load_model_and_data(self) -> Tuple[Any, pd.DataFrame, pd.Series]:
        """Загрузка модели и тестовых данных"""
        experiment_name = self.config['experiment_name']
        model_path = f"models/experiments/{experiment_name}/model.pkl"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Загрузка модели
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Загрузка тестовых данных
        test_df = pd.read_csv("data/features/test_features.csv", index_col=0)
        
        target_col = self.config['data']['target_column']
        date_col = self.config['data']['date_column']
        
        # Колонки для исключения (целевая переменная, дата, id колонки)
        columns_to_exclude = [target_col]
        if date_col in test_df.columns:
            columns_to_exclude.append(date_col)
        
        # Исключаем ID колонки если они есть
        id_columns = [col for col in test_df.columns if 'customer' in col.lower() or 'id' in col.lower()]
        columns_to_exclude.extend(id_columns)
        
        X_test = test_df.drop(columns=columns_to_exclude)
        y_test = test_df[target_col]
        
        self.logger.info(f"Модель загружена: {model_path}")
        self.logger.info(f"Тестовые данные: {X_test.shape}")
        
        return model, X_test, y_test
    
    def calculate_all_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                             y_pred: np.ndarray) -> Dict[str, float]:
        """Расчет всех метрик качества"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            
            # Дополнительные метрики
            'specificity': self._calculate_specificity(y_true, y_pred),
            'balanced_accuracy': self._calculate_balanced_accuracy(y_true, y_pred),
        }
        
        return metrics
    
    def _calculate_specificity(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Расчет специфичности"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_balanced_accuracy(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Расчет сбалансированной точности"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return (sensitivity + specificity) / 2
    
    def find_optimal_threshold(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                              metric: str = 'f1') -> Tuple[float, float]:
        """Поиск оптимального порога для заданной метрики"""
        thresholds = np.arange(0.1, 1.0, 0.01)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced_accuracy':
                score = self._calculate_balanced_accuracy(y_true, y_pred)
            else:
                score = accuracy_score(y_true, y_pred)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, 
                            save_path: str = "reports/figures/confusion_matrix.png"):
        """Построение матрицы ошибок"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted 0', 'Predicted 1'],
                   yticklabels=['Actual 0', 'Actual 1'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Добавляем метрики на график
        tn, fp, fn, tp = cm.ravel()
        plt.text(0.5, -0.1, f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}', 
                ha='center', transform=plt.gca().transAxes)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Логируем в MLflow
        self.mlflow_tracker.log_figure(plt.gcf(), "confusion_matrix.png", "figures")
        
        plt.close()
        self.logger.info(f"Матрица ошибок сохранена: {save_path}")
    
    def plot_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                      save_path: str = "reports/figures/roc_curve.png"):
        """ROC кривая"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Логируем в MLflow
        self.mlflow_tracker.log_figure(plt.gcf(), "roc_curve.png", "figures")
        
        plt.close()
        self.logger.info(f"ROC кривая сохранена: {save_path}")
    
    def plot_precision_recall_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                                   save_path: str = "reports/figures/precision_recall_curve.png"):
        """Precision-Recall кривая"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Baseline (случайный классификатор)
        baseline = y_true.mean()
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Логируем в MLflow
        self.mlflow_tracker.log_figure(plt.gcf(), "precision_recall_curve.png", "figures")
        
        plt.close()
        self.logger.info(f"Precision-Recall кривая сохранена: {save_path}")
    
    def plot_threshold_analysis(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                              save_path: str = "reports/figures/threshold_analysis.png"):
        """Анализ влияния порога на метрики"""
        thresholds = np.arange(0.1, 1.0, 0.01)
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        accuracy_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precision_scores.append(precision_score(y_true, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            accuracy_scores.append(accuracy_score(y_true, y_pred))
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, precision_scores, label='Precision', linewidth=2)
        plt.plot(thresholds, recall_scores, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        plt.plot(thresholds, accuracy_scores, label='Accuracy', linewidth=2)
        
        # Отмечаем оптимальный порог для F1
        best_threshold, best_f1 = self.find_optimal_threshold(y_true, y_pred_proba, 'f1')
        plt.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal F1 threshold: {best_threshold:.3f}')
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Classification Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # Логируем в MLflow
        self.mlflow_tracker.log_figure(plt.gcf(), "threshold_analysis.png", "figures")
        
        plt.close()
        self.logger.info(f"Анализ порогов сохранен: {save_path}")
        
        return best_threshold, best_f1
    
    def generate_classification_report(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Генерация детального отчета по классификации"""
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Добавляем дополнительную информацию
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        report['confusion_matrix'] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }
        
        return report
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Основная функция оценки модели"""
        
        # Начинаем MLflow run для оценки
        self.mlflow_tracker.start_run(
            run_name=f"{self.config['experiment_name']}_evaluation",
            tags={'stage': 'evaluation'}
        )
        
        try:
            # Предсказания
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Поиск оптимального порога
            optimal_threshold, optimal_f1 = self.find_optimal_threshold(y_test, y_pred_proba, 'f1')
            y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Предсказания с дефолтным порогом
            default_threshold = self.config['evaluation']['thresholds']['default']
            y_pred_default = (y_pred_proba >= default_threshold).astype(int)
            
            # Метрики с дефолтным порогом
            default_metrics = self.calculate_all_metrics(y_test, y_pred_proba, y_pred_default)
            
            # Метрики с оптимальным порогом
            optimal_metrics = self.calculate_all_metrics(y_test, y_pred_proba, y_pred_optimal)
            
            # Логируем метрики в MLflow
            default_metrics_prefixed = {f"test_default_{k}": v for k, v in default_metrics.items()}
            optimal_metrics_prefixed = {f"test_optimal_{k}": v for k, v in optimal_metrics.items()}
            
            self.mlflow_tracker.log_metrics(default_metrics_prefixed)
            self.mlflow_tracker.log_metrics(optimal_metrics_prefixed)
            self.mlflow_tracker.log_metrics({
                'optimal_threshold': optimal_threshold,
                'default_threshold': default_threshold
            })
            
            # Создание визуализаций
            self.plot_confusion_matrix(y_test, y_pred_optimal)
            self.plot_roc_curve(y_test, y_pred_proba)
            self.plot_precision_recall_curve(y_test, y_pred_proba)
            best_threshold, best_f1 = self.plot_threshold_analysis(y_test, y_pred_proba)
            
            # Детальный отчет
            classification_rep = self.generate_classification_report(y_test, y_pred_optimal)
            
            # Результаты оценки
            evaluation_results = {
                'test_samples': len(y_test),
                'positive_samples': int(y_test.sum()),
                'negative_samples': int(len(y_test) - y_test.sum()),
                'default_threshold': default_threshold,
                'optimal_threshold': optimal_threshold,
                'default_metrics': default_metrics,
                'optimal_metrics': optimal_metrics,
                'classification_report': classification_rep
            }
            
            # Логируем результаты как JSON
            self.mlflow_tracker.log_dict_as_json(
                evaluation_results, 
                "evaluation_results.json", 
                "evaluation"
            )
            
            # Логируем предсказания
            predictions_df = pd.DataFrame({
                'y_true': y_test,
                'y_pred_proba': y_pred_proba,
                'y_pred_default': y_pred_default,
                'y_pred_optimal': y_pred_optimal
            }, index=X_test.index)
            
            self.mlflow_tracker.log_dataframe(
                predictions_df, 
                "test_predictions.csv", 
                "predictions"
            )
            
            # Завершаем run
            self.mlflow_tracker.end_run()
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Ошибка при оценке модели: {e}")
            self.mlflow_tracker.end_run(status="FAILED")
            raise


def main():
    """Основная функция"""
    # Загрузка конфигурации
    config = load_config()
    
    # Создание evaluator
    evaluator = ModelEvaluator(config)
    
    try:
        # Загрузка модели и данных
        model, X_test, y_test = evaluator.load_model_and_data()
        
        # Оценка модели
        results = evaluator.evaluate_model(model, X_test, y_test)
        
        # Сохранение результатов
        os.makedirs("reports/metrics", exist_ok=True)
        with open("reports/metrics/test_metrics.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Вывод основных метрик
        print(f"Оценка модели завершена")
        print(f"Test AUC (default threshold): {results['default_metrics']['auc']:.4f}")
        print(f"Test F1 (optimal threshold): {results['optimal_metrics']['f1']:.4f}")
        print(f"Optimal threshold: {results['optimal_threshold']:.3f}")
        
    except Exception as e:
        print(f"Ошибка при оценке модели: {e}")
        raise


if __name__ == "__main__":
    main()