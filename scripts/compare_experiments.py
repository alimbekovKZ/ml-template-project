"""
Скрипт для сравнения экспериментов в MLflow.
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import sys
import os

sys.path.append('src')
from utils.config import load_config
from utils.logging import setup_logger


def compare_experiments():
    """Сравнение всех экспериментов"""
    config = load_config()
    logger = setup_logger('CompareExperiments')
    
    # Настройка MLflow
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db')
    mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient()
    experiment_name = config.get('mlflow', {}).get('experiment_name', 'default')
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Эксперимент {experiment_name} не найден. Создаю новый эксперимент.")
            experiment_id = mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment(experiment_id)
            logger.info(f"Создан новый эксперимент: {experiment_name}")
            
            # Если эксперимент только что создан, runs будут отсутствовать
            logger.info("Новый эксперимент не содержит runs. Создание пустого отчета.")
            
            # Создаем пустую таблицу сравнения
            empty_df = pd.DataFrame(columns=[
                'run_id', 'run_name', 'status', 'start_time', 'duration_minutes',
                'model_type', 'val_auc', 'val_f1', 'val_precision', 'val_recall', 
                'test_default_auc', 'learning_rate', 'num_leaves'
            ])
            
            output_path = "reports/experiments_comparison.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            empty_df.to_csv(output_path, index=False)
            logger.info(f"Пустая таблица сравнения сохранена: {output_path}")
            
            print("\n" + "="*100)
            print("СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ")
            print("="*100)
            print(f"\nЭксперимент '{experiment_name}' создан, но пока не содержит runs.")
            print("Запустите обучение моделей для появления данных для сравнения.")
            
            return None
        
        # Получаем все runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.val_auc DESC"]
        )
        
        if not runs:
            logger.info("Runs не найдены в эксперименте. Создание пустого отчета.")
            
            # Создаем пустую таблицу сравнения
            empty_df = pd.DataFrame(columns=[
                'run_id', 'run_name', 'status', 'start_time', 'duration_minutes',
                'model_type', 'val_auc', 'val_f1', 'val_precision', 'val_recall', 
                'test_default_auc', 'learning_rate', 'num_leaves'
            ])
            
            output_path = "reports/experiments_comparison.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            empty_df.to_csv(output_path, index=False)
            logger.info(f"Пустая таблица сравнения сохранена: {output_path}")
            
            print("\n" + "="*100)
            print("СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ")
            print("="*100)
            print(f"\nЭксперимент '{experiment_name}' найден, но не содержит runs.")
            print("Запустите обучение моделей для появления данных для сравнения.")
            
            return
        
        # Создаем DataFrame для сравнения
        comparison_data = []
        
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                'status': run.info.status,
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                'duration_minutes': (run.info.end_time - run.info.start_time) / (1000 * 60) if run.info.end_time else None,
                'model_type': run.data.tags.get('model_type', 'Unknown')
            }
            
            # Добавляем ключевые метрики
            key_metrics = ['val_auc', 'val_f1', 'val_precision', 'val_recall', 'test_default_auc']
            for metric in key_metrics:
                run_data[metric] = run.data.metrics.get(metric, None)
            
            # Добавляем ключевые параметры
            key_params = ['config.model.hyperparameters.learning_rate', 'config.model.hyperparameters.num_leaves']
            for param in key_params:
                param_short = param.split('.')[-1]
                run_data[param_short] = run.data.params.get(param, None)
            
            comparison_data.append(run_data)
        
        # Создаем и выводим сравнительную таблицу
        df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*100)
        print("СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ")
        print("="*100)
        
        # Основная информация
        print(f"\nВсего runs: {len(df)}")
        print(f"Успешных runs: {sum(df['status'] == 'FINISHED')}")
        print(f"Неудачных runs: {sum(df['status'] == 'FAILED')}")
        
        # Топ-5 моделей по validation AUC
        print(f"\nТОП-5 МОДЕЛЕЙ ПО VALIDATION AUC:")
        print("-" * 80)
        
        top_models = df.nlargest(5, 'val_auc')[['run_name', 'model_type', 'val_auc', 'val_f1', 'test_default_auc']]
        print(top_models.to_string(index=False, float_format='%.4f'))
        
        # Статистика по типам моделей
        if 'model_type' in df.columns and df['model_type'].notna().any():
            print(f"\nСТАТИСТИКА ПО ТИПАМ МОДЕЛЕЙ:")
            print("-" * 50)
            
            model_stats = df.groupby('model_type').agg({
                'val_auc': ['count', 'mean', 'std', 'max'],
                'val_f1': ['mean', 'max']
            }).round(4)
            
            print(model_stats)
        
        # Сохраняем детальное сравнение
        output_path = "reports/experiments_comparison.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Детальное сравнение сохранено: {output_path}")
        
        # Лучшая модель
        if not df.empty and df['val_auc'].notna().any():
            best_run = df.loc[df['val_auc'].idxmax()]
            print(f"\nЛУЧШАЯ МОДЕЛЬ:")
            print("-" * 30)
            print(f"Run ID: {best_run['run_id']}")
            print(f"Run Name: {best_run['run_name']}")
            print(f"Model Type: {best_run['model_type']}")
            print(f"Validation AUC: {best_run['val_auc']:.4f}")
            print(f"Validation F1: {best_run['val_f1']:.4f}")
            
            return best_run['run_id']
        
    except Exception as e:
        logger.error(f"Ошибка при сравнении экспериментов: {e}")
        return None


if __name__ == "__main__":
    compare_experiments()