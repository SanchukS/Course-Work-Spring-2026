import joblib
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
from IPython.display import display # Если работаешь в Jupyter
from src.regressor_model_abstract import RegressorModel
from src.metrics_calculation import calculate_metrics

def load_and_evaluate_all_models(
    models_folder: Path | str,
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    window_size: int = 3,
    baselines: Optional[dict[str, RegressorModel]] = None
) -> pd.DataFrame:
    """
    Загружает все модели из папки, объединяет с бейзлайнами, 
    считает все метрики и возвращает единый DataFrame.
    """
    models_to_evaluate = {}
    
    # 1. Добавляем бейзлайны, если они переданы
    if baselines:
        models_to_evaluate.update(baselines)
        
    # 2. Ищем и загружаем все обученные модели (.joblib) из папки
    folder = Path(models_folder)
    if folder.exists():
        for model_path in folder.glob("*.joblib"):
            # Имя модели берем из названия файла
            model_name = model_path.stem 
            print(f"Загрузка модели: {model_name}...")
            models_to_evaluate[model_name] = joblib.load(model_path)
    else:
        print(f"Папка {folder} не найдена!")

    # 3. Рассчитываем метрики
    results_list = []
    print("\nРасчет метрик...")
    for name, model in models_to_evaluate.items():
        metrics = calculate_metrics(model, X_test, y_test, window_size)
        metrics['model_name'] = name
        results_list.append(metrics)
        
    # 4. Формируем DataFrame
    results_df = pd.DataFrame(results_list).set_index('model_name')
    
    return results_df

def display_metrics_by_group(results_df: pd.DataFrame, metric_type: str = 'mape'):
    """
    Удобная функция для просмотра огромной таблицы.
    metric_type: 'mae', 'mase' или 'mape'
    """
    # Выбираем только те колонки, которые заканчиваются на нужную метрику
    filtered_df = results_df.filter(like=f'_{metric_type}')
    
    # Немного причешем названия колонок для красоты вывода
    filtered_df.columns = [c.replace(f'_{metric_type}', '') for c in filtered_df.columns]
    
    print(f"\n{'='*20} СРАВНЕНИЕ ПО {metric_type.upper()} {'='*20}")
    pd.set_option('display.float_format', '{:.4f}'.format)
    display(filtered_df)