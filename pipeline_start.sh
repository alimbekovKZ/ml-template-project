#!/bin/bash

# Скрипт для запуска полного ML пайплайна и фиксации в Git
set -e  # Остановка при ошибке

echo "Запуск ML пайплайна..."

# Загружаем конфигурацию эксперимента
EXPERIMENT_NAME=$(grep "experiment_name:" params.yaml | head -1 | awk '{print $2}' | tr -d '"')

echo "Эксперимент: $EXPERIMENT_NAME"

# Создаем новую ветку для эксперимента
echo "Создание ветки для эксперимента..."
git branch $EXPERIMENT_NAME 2>/dev/null || echo "Ветка $EXPERIMENT_NAME уже существует"
git checkout $EXPERIMENT_NAME

# Запуск полного DVC пайплайна
echo "Запуск DVC пайплайна..."
dvc repro

# Добавляем все DVC файлы в Git
echo "Добавление файлов в Git..."
git add dvc.lock
git add params.yaml
git add *.dvc
git add data/.gitignore
git add models/.gitignore

# Добавляем результаты экспериментов
git add reports/metrics/
git add reports/figures/

# Коммитим изменения
echo "Сохранение эксперимента..."
git commit -m "Эксперимент: $EXPERIMENT_NAME - $(date '+%Y-%m-%d %H:%M:%S')" || echo "Нет изменений для коммита"

# Возвращаемся на main ветку
git checkout main 2>/dev/null || git checkout master 2>/dev/null || echo "Остаемся на текущей ветке"

echo "Пайплайн завершен!"
echo "Ветка эксперимента: $EXPERIMENT_NAME"
echo "Результаты в reports/"

# Показываем статистику
if [ -f "reports/metrics/test_metrics.json" ]; then
    echo "📊 Основные метрики:"
    python3 -c "
import json
try:
    with open('reports/metrics/test_metrics.json', 'r') as f:
        metrics = json.load(f)
    if 'optimal_metrics' in metrics:
        m = metrics['optimal_metrics']
        print(f'  AUC: {m.get(\"auc\", 0):.4f}')
        print(f'  F1:  {m.get(\"f1\", 0):.4f}')
        print(f'  Precision: {m.get(\"precision\", 0):.4f}')
        print(f'  Recall: {m.get(\"recall\", 0):.4f}')
except:
    print('  Метрики недоступны')
"
fi

echo "Готово!"