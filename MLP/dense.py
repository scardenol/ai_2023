import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size, momentum=0):
        # Initialize the weights and bias randomly.
        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, (output_size, 1))

       
        self.delta_weights = 0
        self.delta_bias = 0
        self.momentum = momentum


    def forward(self, input):
        # Compute the output of the layer.
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate, momentum):
        # Compute the local gradient of the loss with respect to the weights and bias for the layer.
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Compute the delta weights and delta bias.
        # Notice that we are using explicitly the momentum term but only if it is different from zero.
        self.delta_weights = self.momentum * self.delta_weights + learning_rate * weights_gradient
        self.delta_bias = self.momentum * self.delta_bias + learning_rate * output_gradient

        # Update the weights and bias.
        self.weights = self.weights - self.delta_weights
        self.bias = self.bias -  self.delta_bias

        # Return the gradient of the loss with respect to the input of the layer.
        return input_gradient