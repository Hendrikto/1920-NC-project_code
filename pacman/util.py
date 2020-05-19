import numpy as np


def ones_at(shape, positions, dtype):
    array = np.zeros(shape, dtype=dtype)
    if positions:
        array[tuple(map(list, zip(*positions)))] = 1
    return array
