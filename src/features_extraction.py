import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import cast

def extract_features(history_df: pd.DataFrame, window_size: int = 3) -> dict[str, float]:
    """
    Генерирует фичи для текущего момента времени на основе прошлой истории.
    history_df: DataFrame с колонками ['DAY', 'QUANTITY', 'WEIGHT_GR', 'VOLUME_ML', 'COUNT'].
                Длина history_df гарантированно >= 2. Самая последняя строка - текущая покупка.
    """
    features = {}
    current_idx = len(history_df)
    
    # 1. Признаки текущего состояния
    current_row = history_df.iloc[-1]
    features['day_of_week'] = int(current_row['DAY'] % 7)
    features['session_step'] = int(current_idx)
    
    features['current_quantity'] = float(current_row['QUANTITY'])
    features['current_weight'] = float(current_row['WEIGHT_GR'])
    features['current_volume'] = float(current_row['VOLUME_ML'])
    features['current_count'] = float(current_row['COUNT'])
    
    # Вычисляем массив всех исторических интервалов
    intervals = history_df['DAY'].diff().dropna().values.astype(float)
    intervals = cast(npt.NDArray[np.float64], intervals)

    # 2. Агрегации и скользящие средние (Считаем до лагов, чтобы использовать для нормализации)
    features['mean_interval'] = float(intervals.mean())
    features['std_interval'] = float(intervals.std()) if len(intervals) > 1 else -1.0
    features['ema_interval'] = float(pd.Series(intervals).ewm(alpha=0.5, adjust=False).mean().iloc[-1])
    
    # Простое скользящее среднее по размеру окна
    window_intervals = intervals[-window_size:] if len(intervals) > 0 else []
    features['rolling_mean_interval'] = float(np.mean(window_intervals)) if len(window_intervals) > 0 else -1.0

    features['mean_quantity'] = float(history_df['QUANTITY'].mean())
    features['mean_weight'] = float(history_df['WEIGHT_GR'].mean())
    features['mean_volume'] = float(history_df['VOLUME_ML'].mean())
    features['mean_count'] = float(history_df['COUNT'].mean())

    # 3. Лаговые признаки и их нормализация
    for i in range(1, window_size + 1):
        row_idx = current_idx - 1 - i
        
        if row_idx >= 0:
            features[f'quantity_lag_{i}'] = float(history_df.iloc[row_idx]['QUANTITY'])
            features[f'weight_lag_{i}'] = float(history_df.iloc[row_idx]['WEIGHT_GR'])
            features[f'volume_lag_{i}'] = float(history_df.iloc[row_idx]['VOLUME_ML'])
            features[f'count_lag_{i}'] = float(history_df.iloc[row_idx]['COUNT'])
        else:
            features[f'quantity_lag_{i}'] = -1.0
            features[f'weight_lag_{i}'] = -1.0
            features[f'volume_lag_{i}'] = -1.0
            features[f'count_lag_{i}'] = -1.0

        interval_idx = len(intervals) - i
        if interval_idx >= 0:
            lag_val = float(intervals[interval_idx])
            features[f'interval_lag_{i}'] = lag_val
            # Нормализованный интервал
            features[f'norm_interval_lag_{i}'] = lag_val / features['mean_interval']
        else:
            features[f'interval_lag_{i}'] = -1.0
            features[f'norm_interval_lag_{i}'] = -1.0

    # 4. Взаимодействия и подсказки для ML-модели
    metrics = ['quantity', 'weight', 'volume', 'count']
    
    for metric in metrics:
        curr_val = features[f'current_{metric}']
        mean_val = features[f'mean_{metric}']
        
        # Отношение текущей покупки к средней исторической
        features[f'{metric}_to_mean_ratio'] = curr_val / mean_val if mean_val > 0 else 1.0
        
        # Математическая проекция ожидаемого интервала
        if mean_val > 0 and features['mean_interval'] > 0:
            features[f'baseline_pred_by_{metric}'] = (features['mean_interval'] / mean_val) * curr_val
        else:
            features[f'baseline_pred_by_{metric}'] = -1.0

    return features