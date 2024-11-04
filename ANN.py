import matplotlib.pyplot as plt
from pynndescent import NNDescent
import numpy as np


def ann(index: NNDescent, vector: np.ndarray, K):
    inputPoint = np.array(vector)
    results = index.query([inputPoint], K)
    return results