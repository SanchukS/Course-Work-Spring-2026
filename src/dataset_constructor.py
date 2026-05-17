import numpy as np
import pandas as pd
from tqdm import tqdm
from src.features_extraction import extract_features

def build_dataset_from_session(session_df: pd.DataFrame, window_size: int = 3) -> tuple[list[dict[str, float]], list[int]]:
    """
    Проходится по одной сессии пользователя и генерирует X и y.
    session_df: DataFrame с колонками ['DAY', 'WEIGHT_GR', 'VOLUME_ML', 'COUNT']
    """
    X_list = []
    y_list = []
    
    # Нам нужно минимум 1 покупка для фичей и 1 следующая для таргета
    for i in range(2, len(session_df)):
        history_df = session_df.iloc[:i]
        target_row = session_df.iloc[i]
        
        features = extract_features(history_df, window_size)
        target = target_row['DAY'] - history_df.iloc[-1]['DAY']
        
        X_list.append(features)
        y_list.append(target)
        
    return X_list, y_list


def build_dataset(outflow_data: pd.DataFrame, window_size: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """
    Проходит по всем сессиям, генерирует по каждой из них X и y, объединяет в один датасет.
    outflow_data: DataFrame с колонками ['TOTAL CATEGORY', 'SESSION_ID', 'DAY', 'WEIGHT_GR', 'VOLUME_ML', 'COUNT']
    """

    grouped_sessions = outflow_data.groupby(["TOTAL CATEGORY", "SESSION_ID"])

    X, y = [], []

    for name, session_df in tqdm(
        grouped_sessions, 
        desc="Building dataset",
        unit="session",
        ncols=100,
        colour="green",
        leave=True
    ):
        try:
            X_session, y_session = build_dataset_from_session(session_df, window_size)
            X.extend(X_session)
            y.extend(y_session)
        except Exception as e:
            tqdm.write(f"Error processing {name}: {e}")
            continue

    X_result = pd.DataFrame(X)
    y_result = pd.Series(y)

    return X_result, y_result