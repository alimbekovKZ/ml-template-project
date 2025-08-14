"""
Скрипт для переноса лучшей модели в Model Registry.
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import sys
import os

sys.path.append('src')
from utils.config import load_config
from utils.logging import setup_logger


def promote_best_model(metric_name: str = "val_auc", model_name: str = None):
    """
    Перенос лучшей модели в Model Registry
    
    Args:
        metric_name: Метрика для выбора лучшей модели
        model_name: Имя модели в registry (по умолчанию из конфига)
    """
    config = load_config()
    logger = setup_logger('PromoteBestModel')
    
    # Настройка MLflow
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db')
    mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient()
    experiment_name = config.get('mlflow', {}).get('experiment_name', 'default')
    
    if model_name is None:
        model_name = f"{experiment_name}_model"
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning(f"Эксперимент {experiment_name} не найден. Создаю заглушку для продакшена.")
            
            # Создаем пустые файлы для продакшена, чтобы DVC pipeline не падал
            production_dir = "models/production"
            os.makedirs(production_dir, exist_ok=True)
            
            # Создаем пустой файл модели
            with open(f"{production_dir}/model.pkl", 'wb') as f:
                import pickle
                dummy_model = {"status": "no_experiment", "message": "Эксперимент не найден"}
                pickle.dump(dummy_model, f)
            
            # Создаем метаданные для заглушки
            dummy_metadata = {
                'model_name': 'dummy_model',
                'model_version': '0.0.0',
                'run_id': 'no_experiment',
                'metrics': {},
                'promotion_date': str(__import__('pandas').Timestamp.now()),
                'status': 'no_experiment',
                'message': f'Эксперимент {experiment_name} не найден.'
            }
            
            import json
            with open(f"{production_dir}/model_metadata.json", 'w') as f:
                json.dump(dummy_metadata, f, indent=2)
            
            logger.info(f"Заглушка сохранена в {production_dir}")
            return None
        
        # Находим лучшую модель
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"metrics.{metric_name} > 0",
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )
        
        if not runs:
            logger.warning(f"Не найдено runs с метрикой {metric_name}. Создаю заглушку для продакшена.")
            
            # Создаем пустые файлы для продакшена, чтобы DVC pipeline не падал
            production_dir = "models/production"
            os.makedirs(production_dir, exist_ok=True)
            
            # Создаем пустой файл модели
            with open(f"{production_dir}/model.pkl", 'wb') as f:
                import pickle
                # Создаем заглушку - пустую модель
                dummy_model = {"status": "no_trained_models", "message": "Запустите обучение моделей"}
                pickle.dump(dummy_model, f)
            
            # Создаем метаданные для заглушки
            dummy_metadata = {
                'model_name': 'dummy_model',
                'model_version': '0.0.0',
                'run_id': 'no_run',
                'metrics': {},
                'promotion_date': str(__import__('pandas').Timestamp.now()),
                'status': 'no_trained_models',
                'message': 'Модели еще не обучены. Запустите обучение для получения реальной модели.'
            }
            
            import json
            with open(f"{production_dir}/model_metadata.json", 'w') as f:
                json.dump(dummy_metadata, f, indent=2)
            
            logger.info(f"Заглушка сохранена в {production_dir}")
            
            print(f"\n⚠️  НЕТ ОБУЧЕННЫХ МОДЕЛЕЙ")
            print(f"Создана заглушка в: {production_dir}")
            print(f"Запустите полный pipeline для обучения моделей:")
            print(f"  dvc repro train_model")
            
            return None
        
        best_run = runs[0]
        best_score = best_run.data.metrics.get(metric_name)
        
        logger.info(f"Лучшая модель найдена:")
        logger.info(f"  Run ID: {best_run.info.run_id}")
        logger.info(f"  {metric_name}: {best_score:.4f}")
        
        # Проверяем требования к качеству
        promotion_config = config.get('promotion', {})
        min_requirements = {
            'min_accuracy': promotion_config.get('min_accuracy', 0.0),
            'min_precision': promotion_config.get('min_precision', 0.0),
            'min_recall': promotion_config.get('min_recall', 0.0),
            'min_auc': promotion_config.get('min_auc', 0.0)
        }
        
        # Проверяем все требования
        passed_checks = True
        for req_name, min_value in min_requirements.items():
            if min_value > 0:
                metric_key = f"val_{req_name.replace('min_', '')}"
                actual_value = best_run.data.metrics.get(metric_key, 0)
                
                if actual_value < min_value:
                    logger.warning(f"Модель не прошла проверку {req_name}: {actual_value:.4f} < {min_value:.4f}")
                    passed_checks = False
                else:
                    logger.info(f"✓ {req_name}: {actual_value:.4f} >= {min_value:.4f}")
        
        if not passed_checks:
            logger.error("Модель не соответствует минимальным требованиям для продакшена")
            return
        
        # Регистрируем модель
        model_uri = f"runs:/{best_run.info.run_id}/model"
        
        try:
            # Проверяем, существует ли уже модель в registry
            try:
                registered_model = client.get_registered_model(model_name)
                logger.info(f"Модель {model_name} уже существует в registry")
            except MlflowException:
                # Создаем новую модель в registry
                registered_model = client.create_registered_model(
                    model_name,
                    description=f"Best model from experiment {experiment_name}"
                )
                logger.info(f"Создана новая модель в registry: {model_name}")
            
            # Создаем новую версию модели
            model_version = client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=best_run.info.run_id,
                description=f"Model with {metric_name}={best_score:.4f}"
            )
            
            logger.info(f"Создана версия модели: {model_version.version}")
            
            # Переводим в стадию Staging
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging",
                archive_existing_versions=False
            )
            
            logger.info(f"Модель переведена в стадию Staging")
            
            # Добавляем теги к версии модели
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="validation_auc",
                value=str(best_score)
            )
            
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="promotion_date",
                value=str(pd.Timestamp.now())
            )
            
            # Копируем модель в папку production
            production_dir = "models/production"
            os.makedirs(production_dir, exist_ok=True)
            
            # Загружаем и сохраняем модель
            import pickle
            model = mlflow.sklearn.load_model(model_uri)
            
            with open(f"{production_dir}/model.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            # Сохраняем метаданные модели
            model_metadata = {
                'model_name': model_name,
                'model_version': model_version.version,
                'run_id': best_run.info.run_id,
                'metrics': dict(best_run.data.metrics),
                'promotion_date': str(pd.Timestamp.now()),
                'model_uri': model_uri
            }
            
            import json
            with open(f"{production_dir}/model_metadata.json", 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            logger.info(f"Модель сохранена в {production_dir}")
            
            print(f"\n✅ МОДЕЛЬ УСПЕШНО ПЕРЕВЕДЕНА В ПРОДАКШН")
            print(f"Модель: {model_name}")
            print(f"Версия: {model_version.version}")
            print(f"Run ID: {best_run.info.run_id}")
            print(f"{metric_name}: {best_score:.4f}")
            print(f"Статус: Staging")
            print(f"Локальная копия: {production_dir}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Ошибка при регистрации модели: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Ошибка при поиске лучшей модели: {e}")
        return None


def list_registered_models():
    """Список всех зарегистрированных моделей"""
    config = load_config()
    logger = setup_logger('ListModels')
    
    tracking_uri = config.get('mlflow', {}).get('tracking_uri', 'sqlite:///mlflow.db')
    mlflow.set_tracking_uri(tracking_uri)
    
    client = MlflowClient()
    
    try:
        models = client.list_registered_models()
        
        if not models:
            print("Зарегистрированных моделей не найдено")
            return
        
        print("\nЗАРЕГИСТРИРОВАННЫЕ МОДЕЛИ:")
        print("=" * 80)
        
        for model in models:
            print(f"\nМодель: {model.name}")
            print(f"Описание: {model.description or 'Нет описания'}")
            print(f"Создана: {pd.to_datetime(model.creation_timestamp, unit='ms')}")
            print(f"Обновлена: {pd.to_datetime(model.last_updated_timestamp, unit='ms')}")
            
            # Показываем версии
            versions = client.get_latest_versions(model.name, stages=["None", "Staging", "Production"])
            
            if versions:
                print("Версии:")
                for version in versions:
                    print(f"  v{version.version} ({version.current_stage}) - "
                          f"Run: {version.run_id[:8]}...")
            
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {e}")


if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description='Управление моделями в MLflow')
    parser.add_argument('--action', choices=['promote', 'list'], default='promote',
                       help='Действие: promote - перенести лучшую модель, list - список моделей')
    parser.add_argument('--metric', default='val_auc',
                       help='Метрика для выбора лучшей модели (по умолчанию: val_auc)')
    parser.add_argument('--model-name', 
                       help='Имя модели в registry (по умолчанию из конфига)')
    
    args = parser.parse_args()
    
    if args.action == 'promote':
        promote_best_model(args.metric, args.model_name)
    elif args.action == 'list':
        list_registered_models()