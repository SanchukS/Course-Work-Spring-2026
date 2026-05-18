import numpy as np
import pandas as pd
from typing import Protocol


class RegressorModel(Protocol):
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray: ...