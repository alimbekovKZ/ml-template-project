# ML Template Project

Шаблонный проект для машинного обучения с использованием DVC, поддерживающий полный цикл разработки ML моделей.

## 🚀 Быстрый старт

```bash
# Клонирование репозитория
git clone <your-repo-url>
cd ml-template-project


# Добавление данных
cp your_dataset.csv data/raw/dataset.csv
dvc add data/raw/dataset.csv
git add data/raw/.gitignore data/raw/dataset.csv.dvc
git commit -m "Add raw dataset"
```

## 📋 Структура проекта

```
ml-template-project/
├── configs/                   # Конфигурации
│   ├── config.yaml           # Основная конфигурация
│   ├── model_configs/        # Конфигурации моделей
│   └── data_configs/         # Конфигурации данных
├── data/
│   ├── raw/                  # Исходные данные (DVC tracked)
│   ├── processed/            # Обработанные данные (DVC tracked)
│   └── features/             # Признаки (DVC tracked)
├── models/
│   ├── experiments/          # Экспериментальные модели
│   └── production/           # Продакшн модели
├── src/                      # Исходный код
│   ├── data/                 # Работа с данными
│   ├── features/             # Feature engineering
│   ├── models/               # Обучение и предсказания
│   ├── evaluation/           # Оценка моделей
│   └── utils/                # Утилиты
├── reports/                  # Отчеты и визуализации
├── dvc.yaml                  # DVC пайплайн
└── params.yaml               # Параметры эксперимента
```

## 🔧 Конфигурация

### Основные параметры (params.yaml)

```yaml
experiment_name: "my_experiment"
random_seed: 42

data:
  source_path: "data/raw/dataset.csv"
  target_column: "target"
  date_column: "date"

model:
  type: "catboost"  # catboost, xgboost
  hyperparameters:
    objective: "binary"
    metric: "auc"
    learning_rate: 0.1
```

### Поддерживаемые модели

- **CatBoost**: Автоматическая работа с категориальными признаками
- **XGBoost**: Классический градиентный бустинг

## 📊 Пайплайн ML

### 1. Загрузка данных
```bash
# Автоматическая загрузка через DVC
python src/data/make_dataset.py
```

### 2. Предобработка
```bash
# Фильтрация, очистка, разбиение на train/val/test
python src/data/preprocessing.py
```

### 3. Feature Engineering
```bash
# Создание признаков, кодирование, масштабирование
python src/features/build_features.py
```

### 4. Обучение модели
```bash
# Обучение с кросс-валидацией и early stopping
python src/models/train_model.py
```

### 5. Оценка модели
```bash
# Расчет метрик и создание визуализаций
python src/evaluation/evaluate_model.py
```

### Запуск полного пайплайна
```bash
# Все этапы одной командой
dvc repro
```

## 🧪 Эксперименты

### Запуск нового эксперимента
```bash
# Изменить параметры в params.yaml, затем:
dvc exp run --name my_experiment_v2
```

### Сравнение экспериментов
```bash
# Просмотр всех экспериментов
dvc exp show

# Сравнение конкретных экспериментов
dvc exp diff experiment_1 experiment_2
```

### Отслеживание метрик
Метрики автоматически сохраняются в `reports/metrics/` и отслеживаются DVC:
- `train_metrics.json` - метрики на обучающей выборке
- `test_metrics.json` - метрики на тестовой выборке

## 📈 Мониторинг и визуализация

### Автоматически создаваемые графики
- Feature importance
- ROC кривая
- Precision-Recall кривая
- Confusion matrix
- Learning curves
- SHAP values

## 📈 MLflow интеграция

### Автоматический трекинг экспериментов

Все метрики, параметры, модели и артефакты автоматически логируются в MLflow:

- **Параметры**: Конфигурация модели, гиперпараметры, настройки данных
- **Метрики**: AUC, F1, Precision, Recall для train/validation/test
- **Модели**: Автоматическое сохранение с подписями и примерами
- **Артефакты**: Графики, feature importance, предсказания, конфигурации

### Запуск MLflow UI
```bash
# Локальный запуск
make mlflow-ui

# Или напрямую
mlflow ui --host 0.0.0.0 --port 5000
```

Откройте http://localhost:5000 для просмотра экспериментов.

### Сравнение экспериментов

Используйте MLflow UI для сравнения экспериментов:
- Откройте http://127.0.0.1:5000/#/experiments/
- Выберите несколько runs для сравнения
- Просматривайте метрики, параметры и артефакты

### Model Registry и продакшн

```bash
# Автоматический перенос лучшей модели
make mlflow-promote

# Ручной перенос с параметрами
python scripts/promote_best_model.py --metric val_f1 --model-name my_model

# Список зарегистрированных моделей
python scripts/promote_best_model.py --action list
```

### Жизненный цикл модели в MLflow

1. **Эксперимент** → Автоматическое логирование всех данных
2. **Сравнение** → Анализ метрик и выбор лучшей модели  
3. **Staging** → Тестирование модели перед продакшном
4. **Production** → Развертывание в боевой среде

### Конфигурация MLflow

```yaml
# params.yaml
mlflow:
  enabled: true
  tracking_uri: "sqlite:///mlflow.db"  # Или удаленный сервер
  experiment_name: "my_project"
  auto_log: true
  log_models: true
  log_artifacts: true
```

### Удаленный MLflow сервер

```bash
# Для использования удаленного сервера
export MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Или в params.yaml
mlflow:
  tracking_uri: "http://mlflow-server:5000"
```


## 📖 Документация

### Jupyter notebooks
- `notebooks/exploratory/` - исследовательский анализ
- `notebooks/reports/` - отчеты и презентации

## 🔧 Настройка для вашего проекта

### 1. Обновите конфигурацию
```yaml
# params.yaml
experiment_name: "your_project_name"
data:
  source_path: "path/to/your/data.csv"
  target_column: "your_target"
  date_column: "your_date_column"
```

### 2. Адаптируйте код под ваши данные
- Обновите `src/data/preprocessing.py` под ваши фильтры
- Настройте `src/features/build_features.py` под ваши признаки
- Измените метрики в `src/evaluation/evaluate_model.py`


**Примечание**: Этот шаблон создан для ускорения разработки ML проектов. Адаптируйте его под свои потребности и требования проекта.






