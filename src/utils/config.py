"""
Утилиты для работы с конфигурацией проекта.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = "params.yaml") -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML файла.
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        Словарь с конфигурацией
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Обработка переменных окружения
    config = _process_env_variables(config)
    
    return config


def _process_env_variables(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обработка переменных окружения в конфигурации.
    Заменяет значения вида ${VAR_NAME} на значения из переменных окружения.
    """
    def replace_env_vars(obj):
        if isinstance(obj, dict):
            return {key: replace_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj
    
    return replace_env_vars(config)


def save_config(config: Dict[str, Any], config_path: str = "params.yaml"):
    """
    Сохранение конфигурации в YAML файл.
    
    Args:
        config: Словарь с конфигурацией
        config_path: Путь для сохранения
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Загрузка конфигурации для конкретного типа модели.
    
    Args:
        model_type: Тип модели (lightgbm, catboost, xgboost)
        
    Returns:
        Конфигурация модели
    """
    config_path = f"configs/model_configs/{model_type}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурация модели не найдена: {config_path}")
    
    return load_config(config_path)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Валидация конфигурации проекта.
    
    Args:
        config: Конфигурация для валидации
        
    Returns:
        True если конфигурация валидна
        
    Raises:
        ValueError: Если конфигурация невалидна
    """
    required_sections = ['data', 'preprocessing', 'features', 'model', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Отсутствует обязательная секция: {section}")
    
    # Валидация секции data
    data_config = config['data']
    required_data_fields = ['target_column', 'date_column']
    for field in required_data_fields:
        if field not in data_config:
            raise ValueError(f"Отсутствует обязательное поле в секции data: {field}")
    
    # Валидация секции model
    model_config = config['model']
    if 'type' not in model_config:
        raise ValueError("Не указан тип модели")
    
    supported_models = ['lightgbm', 'catboost', 'xgboost']
    if model_config['type'] not in supported_models:
        raise ValueError(f"Неподдерживаемый тип модели: {model_config['type']}. "
                        f"Поддерживаемые: {supported_models}")
    
    return True


def update_config_from_args(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обновление конфигурации из аргументов командной строки.
    
    Args:
        config: Базовая конфигурация
        args: Аргументы для обновления
        
    Returns:
        Обновленная конфигурация
    """
    updated_config = config.copy()
    
    for key, value in args.items():
        if '.' in key:
            # Поддержка вложенных ключей типа "model.learning_rate"
            keys = key.split('.')
            current = updated_config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            updated_config[key] = value
    
    return updated_config


def get_project_root() -> Path:
    """
    Получение корневой директории проекта.
    
    Returns:
        Путь к корневой директории
    """
    current_file = Path(__file__)
    
    # Ищем директорию с dvc.yaml или .git
    for parent in current_file.parents:
        if (parent / 'dvc.yaml').exists() or (parent / '.git').exists():
            return parent
    
    # Если не найдено, возвращаем текущую директорию
    return Path.cwd()


def setup_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Настройка путей в конфигурации относительно корня проекта.
    
    Args:
        config: Конфигурация
        
    Returns:
        Конфигурация с обновленными путями
    """
    project_root = get_project_root()
    
    # Пути, которые нужно сделать абсолютными
    path_fields = [
        'data.source_path',
        'logging.log_file'
    ]
    
    for path_field in path_fields:
        keys = path_field.split('.')
        current = config
        
        # Навигация к нужному полю
        for key in keys[:-1]:
            if key in current:
                current = current[key]
            else:
                break
        else:
            # Обновляем путь если поле существует
            last_key = keys[-1]
            if last_key in current and isinstance(current[last_key], str):
                current[last_key] = str(project_root / current[last_key])
    
    return config