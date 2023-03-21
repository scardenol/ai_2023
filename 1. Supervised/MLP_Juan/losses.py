
import numpy as np

def instantEnergy(d, y):
    return 0.5 * np.power(d - y.reshape(d.shape), 2).sum()

def instantEnergy_prime(d, y):
    return y - d.reshape(y.shape)

def mse(d, y):
    return np.mean(np.power(d - y, 2))

def mse_prime(d, y):
    return 2 * (y - d) / np.size(d)

def binary_cross_entropy(d, y):
    return np.mean(-d * np.log(y) - (1 - d) * np.log(1 - y))

def binary_cross_entropy_prime(d, y):
    return ((1 - d) / (1 - y) - d / y) / np.size(d)