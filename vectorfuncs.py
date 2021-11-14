import numpy as np
from math import sqrt
def distance(vect1: np.array, vect2: np.array) -> float:
    """
        Calculates Euclidean distance on two vectors of arbitrary length.
    """
    return abs(difference(vect1, vect2))

def difference(vect1: np.array, vect2: np.array) -> np.array:
    """
        Calculate the difference between two vectors
        TODO: ensure shape
    """
    assert len(vect1) == len(vect2), "Vectors must be of the same length."
    return vect1 - vect2

def abs(vect: np.array) -> np.array:
    """
        Calculate the absolute value (length) of the vector 
        TODO: ensure shape
    """
    return sqrt(np.sum(vect ** 2, axis = 0))

def average(vect: np.array) -> np.array:
    """
        Calculate the average point of an array of points.
        TODO: ensure shape
    """
    return np.sum(vect, axis = 0) / len(vect)

def gate(vect: np.array, minimum: float) -> np.array:
    """
        If the size (det. by abs) is less than `minimum`, then it will return the 0 vector
    """
    return vect if abs(vect) >= minimum else np.zeros(vect.shape)

def clamp(vect: np.array, min, max, inplace = True) -> np.array:
    """
        Clamp an array between `min` and `max`. Note that this MODIFIES, and doesn't copy, BY DEFAULT (`inplace`)
    """
    assert len(vect.shape) == 1 and len(vect) == len(min) and len(vect) == len(max), "Shape mismatch!"
    if inplace:
        for v in range(len(vect)):
            vect[v] = min[v] if vect[v] < min[v] else vect[v]
            vect[v] = max[v] if vect[v] > max[v] else vect[v]

    else:
        res = np.zeros(vect.shape)
        for v in range(len(vect)):
            res[v] = min[v] if vect[v] < min[v] else vect[v]
            res[v] = max[v] if vect[v] > max[v] else vect[v]
        return res