import math
import numpy as np
from numba import njit

@njit
def get_best_child_and_ucb(win_rates, visit_num, c_param):
    best_index = -1
    best_value = -np.inf
    
    for i in range(len(win_rates)):
        win_rate = win_rates[i]
        ucb_value = (1 - c_param) * win_rate + c_param * math.sqrt((math.log(visit_num) / visit_num))
        if ucb_value > best_value:
            best_value = ucb_value
            best_index = i

    return best_index