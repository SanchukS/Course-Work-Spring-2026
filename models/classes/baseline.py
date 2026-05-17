import numpy as np
import pandas as pd
from typing import Self
from sklearn.base import RegressorMixin
from src.metrics_calculation import calculate_metrics

class BaselineModel(RegressorMixin):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(X['baseline_pred_count'].values)
