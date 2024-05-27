import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def relu(x):
    return np.maximum(0, x)

