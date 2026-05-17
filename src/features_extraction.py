import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import cast

def extract_features(history_df: pd.DataFrame, window_size: int = 3) -> dict[str, float]:
    """
    Генерирует фичи для текущего момента времени на основе прошлой истории.
    history_df: DataFrame с колонками ['DAY', 'WEIGHT_GR', 'VOLUME_ML', 'COUNT'].
                Длина history_df гарантированно >= 2. Самая последняя строка - текущая покупка.
    """
    features = {}
    current_idx = len(history_df)
    
    # 1. Признаки текущего состояния (текущая транзакция)
    current_row = history_df.iloc[-1]
    features['day_of_week'] = float(current_row['DAY'] % 7)
    features['session_step'] = float(current_idx)
    
    features['current_weight'] = float(current_row['WEIGHT_GR'])
    features['current_volume'] = float(current_row['VOLUME_ML'])
    features['current_count'] = float(current_row['COUNT'])
    
    # Вычисляем массив всех исторических интервалов
    intervals = history_df['DAY'].diff().dropna().values.astype(float)
    intervals = cast(npt.NDArray[np.float64], intervals)

    # 2. Лаговые признаки (начинаем строго с прошлого шага)
    for i in range(1, window_size + 1):
        # Индекс для объемов (текущая - 1 - i)
        row_idx = current_idx - 1 - i
        
        if row_idx >= 0:
            features[f'weight_lag_{i}'] = float(history_df.iloc[row_idx]['WEIGHT_GR'])
            features[f'volume_lag_{i}'] = float(history_df.iloc[row_idx]['VOLUME_ML'])
            features[f'count_lag_{i}'] = float(history_df.iloc[row_idx]['COUNT'])
        else:
            features[f'weight_lag_{i}'] = -1.0
            features[f'volume_lag_{i}'] = -1.0
            features[f'count_lag_{i}'] = -1.0

        # Индекс для интервалов
        interval_idx = len(intervals) - i
        if interval_idx >= 0:
            features[f'interval_lag_{i}'] = float(intervals[interval_idx])
        else:
            features[f'interval_lag_{i}'] = -1.0

    # 3. Агрегации и скользящие средние
    features['mean_interval'] = float(intervals.mean())
    features['std_interval'] = float(intervals.std()) if len(intervals) > 1 else -1.0
    features['ema_interval'] = float(pd.Series(intervals).ewm(alpha=0.5, adjust=False).mean().iloc[-1])

    features['mean_weight'] = float(history_df['WEIGHT_GR'].mean())
    features['mean_volume'] = float(history_df['VOLUME_ML'].mean())
    features['mean_count'] = float(history_df['COUNT'].mean())

    # 4. Взаимодействия и Бейзлайн
    features['count_to_mean_ratio'] = features['current_count'] / features['mean_count'] if features['mean_count'] > 0 else 1.0
    features['weight_to_mean_ratio'] = features['current_weight'] / features['mean_weight'] if features['mean_weight'] > 0 else 1.0

    # Фича Бейзлайна
    features['baseline_pred_count'] = (features['mean_interval'] / features['mean_count']) * features['current_count']

    return features