import numpy as np
from activation import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.power(np.tanh(x), 2)
            
        # Initialize the activation function and its derivative.
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        # Initialize the activation function and its derivative.
        super().__init__(sigmoid, sigmoid_prime)

class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return 1 * np.greater_equal(x, 0)

        # Initialize the activation function and its derivative.
        super().__init__(relu, relu_prime)

class Ramp(Activation):
    def __init__(self):
        # y = ax + b if |y| <= c
        a = 1
        b = 0.5
        c = 1
        def ramp(x):
            return np.maximum(np.minimum(a*x + b, c), -c)
        
        def ramp_prime(x):
            return a * np.less_equal(np.abs(x), c)

        # Initialize the activation function and its derivative.
        super().__init__(ramp, ramp_prime)

class Softmax(Activation):
    def __init__(self):
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / np.sum(e_x, axis=-1, keepdims=True)
        
        def softmax_prime(x):
            s = softmax(x)
            return s * (1 - s)
        # Initialize the activation function and its derivative.
        super().__init__(softmax, softmax_prime)