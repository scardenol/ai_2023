import numpy as np
from layer import Layer

class Activation(Layer):
    # Parent class for all activation functions.
    # It defines the interface that all activation functions must implement.
    def __init__(self, activation, activation_prime):
        # TODO: initialize the activation function and its derivative.
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        # Perform the forward pass.
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate, momentum):
        # Perform the backward pass.
        return np.multiply(output_gradient, self.activation_prime(self.input))