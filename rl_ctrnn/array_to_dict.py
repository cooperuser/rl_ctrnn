import numpy as np

ALPHABET = "abcdefghijklmnopqrtuvwxyz"


def array_to_dict(array: np.ndarray, keys: str = ALPHABET) -> dict:
    if len(array.shape) == 1:
        return {keys[i]: v for i, v in enumerate(array)}
    return {keys[i]: array_to_dict(v, keys) for i, v in enumerate(array)}
