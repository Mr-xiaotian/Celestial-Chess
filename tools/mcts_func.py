import math
import numpy as np
from numba import njit

@njit
def get_best_index(rates_visits, parent_visit: int, c_param: float):
    best_index = -1
    best_value = -np.inf
    
    for i in range(len(rates_visits)):
        win_rate, child_visit = rates_visits[i]
        ucb_value = (1 - c_param) * win_rate + c_param * math.sqrt((math.log(parent_visit) / child_visit))
        if ucb_value > best_value:
            best_value = ucb_value
            best_index = i

    return best_index

@njit
def safe_divide(numerator: float, denominator: int) -> float:
    """安全地计算除法"""
    return numerator / denominator if denominator > 0 else 0