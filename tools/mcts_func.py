import math
import numpy as np
from numba import njit

@njit(cache=True)
def get_best_index_by_ucb1(rates_visits, parent_visit: int, c_param: float):
    best_index = -1
    best_value = -np.inf
    
    for index, (win_rate, child_visit) in enumerate(rates_visits):
        ucb_value = (1 - c_param) * win_rate + c_param * math.sqrt((2 * math.log(parent_visit) / child_visit))
        if ucb_value > best_value:
            best_value = ucb_value
            best_index = index

    return best_index


@njit(cache=True)
def get_best_index_by_puct(rates_visits_probs, parent_visit: int, c_param: float):
    best_index = -1
    best_value = -np.inf
    
    for index, (win_rate, child_visit, policy_prob) in enumerate(rates_visits_probs):
        puct_value = (1 - c_param) * win_rate + c_param * policy_prob * math.sqrt(parent_visit) / (1 + child_visit)

        if puct_value > best_value:
            best_value = puct_value
            best_index = index

    return best_index

