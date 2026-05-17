import numpy as np
import pandas as pd
from math import floor, isclose
from itertools import accumulate
from tqdm import tqdm
from src.dataset_constructor import build_dataset



def build_multiple_datasets(
        outflow_data: pd.DataFrame, 
        window_size: int = 3, 
        datasets_proportions: list[float] = [0.7, 0.15, 0.15], 
        random_state: int = 42
    ) -> list[tuple[pd.DataFrame, pd.Series]]:
    """
    Разбивает исходные данные на N частей по сессиям в соответствии с указанными пропорциями и из каждой части создаёт датасет.
    outflow_data: DataFrame с колонками ['TOTAL CATEGORY', 'SESSION_ID', 'DAY', 'WEIGHT_GR', 'VOLUME_ML', 'COUNT']
    datasets_proportions: Список положительных вещественных чисел суммирующихся в 1.0. Каждое число - доля объектов в соответствующем датасете. 
    """
    assert isclose(sum(datasets_proportions), 1), "Proportions must summrize to 1.0"
    np.random.seed(random_state)

    groups = outflow_data.groupby(["TOTAL CATEGORY", "household_key", "SESSION_ID"])
    all_sessions = list(groups.indices.values())

    shuffled_idexes = np.random.permutation(len(all_sessions))
    sessions_indexes = [all_sessions[i] for i in shuffled_idexes]
    
    len_ = len(sessions_indexes)

    datasets = []

    for idx, (left_prop, right_prop) in enumerate(zip(accumulate([0] + datasets_proportions), accumulate(datasets_proportions)), 1):
        tqdm.write(f"Building dataset: {idx}")
        left_idx = floor(len_ * left_prop)
        right_idx = floor(len_ * right_prop)
        curr_outflow_data = outflow_data.iloc[np.hstack(sessions_indexes[left_idx:right_idx])]

        curr_X, curr_y = build_dataset(curr_outflow_data, window_size)
        datasets.append((curr_X, curr_y))

    return datasets