import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Protocol
from src.regressor_model_abstract import RegressorModel


def calculate_metrics(model: RegressorModel, X_test: pd.DataFrame, y_test: pd.Series, window_size: int = 3) -> dict[str, float]:
    """
    Рассчитывает метрики MAE, MASE и MAPE для overall, cold_start и warm_state,
    включая разбивку каждого из этих состояний на когорты по длине интервала (Fast, Medium, Long).
    """
    # 1. Получаем предсказания модели
    y_pred = model.predict(X_test)
    
    # 2. Базовые маски (Cold Start и Warm State)
    cold_idx = X_test['session_step'] <= window_size
    warm_idx = ~cold_idx
    
    # 3. Маски для когорт по длине интервала
    fast_idx = fast_idx = X_test['mean_interval'] <= 7
    long_idx = fast_idx = X_test['mean_interval'] > 21
    medium_idx = (~fast_idx) & (~long_idx)
    
    # 4. Вспомогательная функция для расчета метрик по любому срезу
    def get_slice_metrics(mask, prefix: str) -> dict[str, float]:
        if mask is None:
            y_t = y_test
            y_p = y_pred
            x_naive = np.array(X_test['interval_lag_1'])
        else:
            # Защита: если в данном срезе нет наблюдений
            if mask.sum() == 0:
                return {f'{prefix}_mae': np.nan, f'{prefix}_mase': np.nan, f'{prefix}_mape': np.nan}
            
            y_t = y_test[mask]
            # .values нужен, чтобы безопасно фильтровать numpy array с помощью pandas Series маски
            y_p = y_pred[mask.values] if isinstance(y_pred, np.ndarray) else y_pred[mask]
            x_naive = np.array(X_test.loc[mask, 'interval_lag_1'])
            
        # -- MAE --
        mae = mean_absolute_error(y_t, y_p)
        
        # -- MASE --
        naive_mae = mean_absolute_error(y_t, x_naive)
        naive_mae = max(naive_mae, 1e-6)  # Защита от деления на ноль
        mase = mae / naive_mae
        
        # -- MAPE --
        # np.maximum используется для защиты от деления на 0, если y_test = 0
        mape = np.mean(np.abs((y_t - y_p) / np.maximum(y_t, 1e-6)))
        
        return {f'{prefix}_mae': mae, f'{prefix}_mase': mase, f'{prefix}_mape': mape}

    # 5. Собираем итоговый словарь с пересечениями масок
    metrics = {}
    
    # --- OVERALL (всё вместе + 3 когорты) ---
    metrics.update(get_slice_metrics(None, 'overall'))
    metrics.update(get_slice_metrics(fast_idx, 'overall_fast'))
    metrics.update(get_slice_metrics(medium_idx, 'overall_medium'))
    metrics.update(get_slice_metrics(long_idx, 'overall_long'))
    
    # --- COLD START (всё вместе + 3 когорты) ---
    metrics.update(get_slice_metrics(cold_idx, 'cold_start'))
    metrics.update(get_slice_metrics(cold_idx & fast_idx, 'cold_start_fast'))
    metrics.update(get_slice_metrics(cold_idx & medium_idx, 'cold_start_medium'))
    metrics.update(get_slice_metrics(cold_idx & long_idx, 'cold_start_long'))
    
    # --- WARM STATE (всё вместе + 3 когорты) ---
    metrics.update(get_slice_metrics(warm_idx, 'warm_state'))
    metrics.update(get_slice_metrics(warm_idx & fast_idx, 'warm_state_fast'))
    metrics.update(get_slice_metrics(warm_idx & medium_idx, 'warm_state_medium'))
    metrics.update(get_slice_metrics(warm_idx & long_idx, 'warm_state_long'))
        
    return metrics