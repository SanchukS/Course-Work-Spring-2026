import numpy as np
import pandas as pd

class QuantityBaselineModel:
    """Бейзлайн 1: Средний интервал на единицу QUANTITY * текущее QUANTITY"""
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array((X['mean_interval'] / X['mean_quantity']) * X['current_quantity'])

class MeanIntervalBaselineModel:
    """Бейзлайн 2: Простое среднее по историческим интервалам"""
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(X["mean_interval"])

class EmaIntervalBaselineModel:
    """Бейзлайн 3: Экспоненциальное скользящее среднее (EMA) по интервалам"""
def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(X['ema_interval'])