import copy
import numpy as np


def custom_deepcopy(obj, _visited=None):
    if _visited is None:
        _visited = {}

    obj_id = id(obj)
    if obj_id in _visited:
        return _visited[obj_id]

    if isinstance(obj, dict):
        copied_obj = {k: custom_deepcopy(v, _visited) for k, v in obj.items()}
    elif isinstance(obj, list):
        copied_obj = [custom_deepcopy(elem, _visited) for elem in obj]
    elif isinstance(obj, set):
        copied_obj = {custom_deepcopy(elem, _visited) for elem in obj}
    elif isinstance(obj, tuple):
        copied_obj = tuple(custom_deepcopy(elem, _visited) for elem in obj)
    elif isinstance(obj, np.ndarray):
        copied_obj = np.copy(obj)
    else:
        copied_obj = copy.copy(obj)

    _visited[obj_id] = copied_obj
    return copied_obj
