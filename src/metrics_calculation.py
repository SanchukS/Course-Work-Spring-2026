import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Protocol

class RegressorModel(Protocol):
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...

def calculate_metrics(model: RegressorModel, X_test: pd.DataFrame, y_test: pd.Series, window_size: int = 3) -> dict[str, float]:
    """
    Рассчитывает метрики MAE и MASE для всей выборки, а также отдельно для Cold Start и Warm State.
    """
    # 1. Получаем предсказания модели
    y_pred = model.predict(X_test)
    
    # 2. Разделяем индексы на Cold Start (прогрев окна) и Warm State (полные данные)
    cold_idx = X_test['session_step'] <= window_size
    warm_idx = ~cold_idx
    
    # 3. Рассчитываем знаменатели для MASE (Ошибка наивного прогноза)
    # Наивный прогноз = последний известный интервал (interval_lag_1). Он всегда существует.
    naive_mae_denominator = mean_absolute_error(y_test, X_test['interval_lag_1'])
    
    if cold_idx.sum() > 0:
        cold_naive_mae_denominator = mean_absolute_error(y_test[cold_idx], X_test.loc[cold_idx, 'interval_lag_1'])
    else:
        cold_naive_mae_denominator = 1e-6
        
    if warm_idx.sum() > 0:
        warm_naive_mae_denominator = mean_absolute_error(y_test[warm_idx], X_test.loc[warm_idx, 'interval_lag_1'])
    else:
        warm_naive_mae_denominator = 1e-6
        
    # Защита от деления на ноль
    naive_mae_denominator = max(naive_mae_denominator, 1e-6)
    cold_naive_mae_denominator = max(cold_naive_mae_denominator, 1e-6)
    warm_naive_mae_denominator = max(warm_naive_mae_denominator, 1e-6)
        
    # 4. Собираем итоговый словарь с метриками
    metrics = {}
    
    # -- Overall --
    metrics['overall_mae'] = mean_absolute_error(y_test, y_pred)
    metrics['overall_mase'] = metrics['overall_mae'] / naive_mae_denominator
    
    # -- Cold Start --
    if cold_idx.sum() > 0:
        metrics['cold_start_mae'] = mean_absolute_error(y_test[cold_idx], y_pred[cold_idx])
        metrics['cold_start_mase'] = metrics['cold_start_mae'] / cold_naive_mae_denominator
    else:
        metrics['cold_start_mae'] = np.nan
        metrics['cold_start_mase'] = np.nan
        
    # -- Warm State --
    if warm_idx.sum() > 0:
        metrics['warm_state_mae'] = mean_absolute_error(y_test[warm_idx], y_pred[warm_idx])
        metrics['warm_state_mase'] = metrics['warm_state_mae'] / warm_naive_mae_denominator
    else:
        metrics['warm_state_mae'] = np.nan
        metrics['warm_state_mase'] = np.nan
        
    return metrics