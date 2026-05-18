import numpy as np
import pandas as pd
import typing as tp
from catboost import CatBoostRegressor
from src.regressor_model_abstract import RegressorModel

class CatBoostRegressorRouting(RegressorModel):
    def __init__(self, catboost_params: dict[str, tp.Any], fast_border: int = 7, long_border: int = 21) -> None:
        self.fast_border = fast_border
        self.long_border = long_border

        self.fast_model = CatBoostRegressor(**catboost_params)
        self.medium_model = CatBoostRegressor(**catboost_params)
        self.long_model = CatBoostRegressor(**catboost_params)

    def _get_gaussian_weights(self, mean_intervals: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует плавные веса для трех моделей на основе Гауссова распределения (Колокол).
        """
        # Определяем "идеальные центры" для каждой модели
        fast_center = self.fast_border / 2.0  # Например, 3.5
        medium_center = (self.fast_border + self.long_border) / 2.0  # Например, 14.0
        long_center = self.long_border + (medium_center - fast_center)  # Симметрично, например 28.5
        
        # Определяем ширину колокола (sigma)
        sigma = (self.long_border - self.fast_border) / 2.0
        
        # Считаем веса (чем дальше от центра, тем меньше вес)
        w_fast = np.exp(-0.5 * ((mean_intervals - fast_center) / sigma)**2)
        w_medium = np.exp(-0.5 * ((mean_intervals - medium_center) / sigma)**2)
        w_long = np.exp(-0.5 * ((mean_intervals - long_center) / sigma)**2)
        
        # Маленькая оптимизация: для краев делаем вес строго = 1.0, 
        # чтобы модели не теряли уверенность на экстремальных значениях
        w_fast[mean_intervals < fast_center] = 1.0
        w_long[mean_intervals > long_center] = 1.0
        
        return w_fast.values, w_medium.values, w_long.values

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **fit_params: tp.Any) -> tp.Self:
        # 1. Считаем мягкие (плавные) веса для ВСЕЙ обучающей выборки
        w_fast, w_medium, w_long = self._get_gaussian_weights(X_train["mean_interval"])

        # Перехватываем и удаляем eval_set и sample_weight из kwargs, чтобы не сломать CatBoost
        eval_set = fit_params.pop("eval_set", None)
        _ = fit_params.pop("sample_weight", None) # Игнорируем внешние веса

        # 2. Режем eval_set ЖЕСТКО (потому что мы хотим делать Early Stopping 
        # именно по тем данным, с которыми модель столкнется на инференсе)
        if eval_set is not None:
            X_val, _ = eval_set
            val_fast_mask = X_val["mean_interval"] <= self.fast_border
            val_long_mask = X_val["mean_interval"] > self.long_border
            val_medium_mask = (~val_fast_mask) & (~val_long_mask)
            
            fast_eval = (X_val[val_fast_mask], eval_set[1][val_fast_mask])
            medium_eval = (X_val[val_medium_mask], eval_set[1][val_medium_mask])
            long_eval = (X_val[val_long_mask], eval_set[1][val_long_mask])
        else:
            fast_eval = medium_eval = long_eval = None

        # 3. ОБУЧЕНИЕ: Все модели видят ВСЕ данные (X_train), но с РАЗНЫМИ весами!
        self.fast_model.fit(X_train, y_train, sample_weight=w_fast, eval_set=fast_eval, **fit_params)
        self.medium_model.fit(X_train, y_train, sample_weight=w_medium, eval_set=medium_eval, **fit_params)
        self.long_model.fit(X_train, y_train, sample_weight=w_long, eval_set=long_eval, **fit_params)

        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # ИНФЕРЕНС: строгое разделение
        fast_mask = X["mean_interval"] <= self.fast_border
        long_mask = X["mean_interval"] > self.long_border
        medium_mask = (~fast_mask) & (~long_mask)

        result = np.zeros(X.shape[0])

        if fast_mask.any():
            result[fast_mask] = self.fast_model.predict(X[fast_mask])
        if medium_mask.any():
            result[medium_mask] = self.medium_model.predict(X[medium_mask])
        if long_mask.any():
            result[long_mask] = self.long_model.predict(X[long_mask])

        return result