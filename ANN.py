import matplotlib.pyplot as plt
from pynndescent import NNDescent
import numpy as np


def ann(data: np.ndarray, vector: np.ndarray, K):
    inputPoint = np.array(vector)
    index = NNDescent(data, leaf_size=K, metric='manhattan')

    results = index.query([inputPoint], K)

    return results