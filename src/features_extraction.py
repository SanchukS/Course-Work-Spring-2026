import numpy as np
import pandas as pd

def extract_features(history_df: pd.DataFrame, window_size: int = 3) -> dict[str, float]:
    """
    Генерирует фичи для текущего момента времени на основе прошлой истории.
    history_df: DataFrame с колонками ['DAY', 'WEIGHT_GR', 'VOLUME_ML', 'COUNT']
                Самая последняя строка - это текущая покупка.
    """
    features = {}
    current_idx = len(history_df)
    
    # 1. Признаки текущего состояния
    current_row = history_df.iloc[-1]
    features['day_of_week'] = current_row['DAY'] % 7
    features['session_step'] = current_idx
    
    # Вычисляем массив всех исторических интервалов в переданном окне
    if current_idx > 1:
        intervals = history_df['DAY'].diff().dropna().values
    else:
        intervals = np.array([])

    # 2. Лаговые признаки (WINDOW_SIZE)
    for i in range(window_size):
        row_idx = current_idx - 1 - i
        
        # Лаги объемов
        if row_idx >= 0:
            features[f'weight_lag_{i}'] = history_df.iloc[row_idx]['WEIGHT_GR']
            features[f'volume_lag_{i}'] = history_df.iloc[row_idx]['VOLUME_ML']
            features[f'count_lag_{i}'] = history_df.iloc[row_idx]['COUNT']
        else:
            features[f'weight_lag_{i}'] = -1
            features[f'volume_lag_{i}'] = -1
            features[f'count_lag_{i}'] = -1

        # Лаги интервалов (интервал_lag_1 = сколько дней прошло ПЕРЕД текущей покупкой)
        interval_idx = len(intervals) - 1 - i
        if interval_idx >= 0:
            features[f'interval_lag_{i+1}'] = intervals[interval_idx]
        else:
            features[f'interval_lag_{i+1}'] = -1

    # 3. Агрегации и скользящие средние (по всей доступной истории)
    if len(intervals) > 0:
        features['mean_interval'] = np.mean(intervals)
        features['std_interval'] = np.std(intervals) if len(intervals) > 1 else -1
        features['ema_interval'] = pd.Series(intervals).ewm(alpha=0.5, adjust=False).mean().iloc[-1]
    else:
        features['mean_interval'] = -1
        features['std_interval'] = -1
        features['ema_interval'] = -1

    features['mean_weight'] = history_df['WEIGHT_GR'].mean()
    features['mean_volume'] = history_df['VOLUME_ML'].mean()
    features['mean_count'] = history_df['COUNT'].mean()

    # 4. Взаимодействия и Бейзлайн
    # Отношение текущей покупки к средней исторической
    features['count_to_mean_ratio'] = features['count_lag_0'] / features['mean_count'] if features['mean_count'] > 0 else 1.0
    features['weight_to_mean_ratio'] = features['weight_lag_0'] / features['mean_weight'] if features['mean_weight'] > 0 else 1.0

    # Фича Бейзлайна: (Средний интервал / Среднее кол-во) * Текущее кол-во
    if features['mean_count'] > 0 and features['mean_interval'] > 0:
        features['baseline_pred_count'] = (features['mean_interval'] / features['mean_count']) * features['count_lag_0']
    else:
        features['baseline_pred_count'] = -1

    return features
