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
