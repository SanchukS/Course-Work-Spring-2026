import numpy as np
import pandas as pd

class QuantityBaselineModel:
    """Бейзлайн 1: Средний интервал на единицу QUANTITY * текущее QUANTITY"""
    def __init__(self, global_fallback: float = 14.0):
        self.global_fallback = global_fallback
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array((X['mean_interval'] / X['mean_quantity']) * X['current_quantity'])

class MeanIntervalBaselineModel:
    """Бейзлайн 2: Простое среднее по историческим интервалам"""
    def __init__(self, global_fallback: float = 14.0):
        self.global_fallback = global_fallback
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(X["mean_interval"])

class EmaIntervalBaselineModel:
    """Бейзлайн 3: Экспоненциальное скользящее среднее (EMA) по интервалам"""
    def __init__(self, global_fallback: float = 14.0):
        self.global_fallback = global_fallback
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(X['ema_interval'])