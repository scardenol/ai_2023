from dense import Dense
from activations import Tanh, Sigmoid, ReLU, Ramp, Softmax
from losses import mse, mse_prime, instantEnergy, instantEnergy_prime
import numpy as np
import time
import matplotlib.pyplot as plt


class NeuralNetwork():
    def __init__(
            self, 
            X: np.ndarray,
            d: np.ndarray,
            n_hiddenLayers: int or list, 
            n_hiddenNeurons:int or list, 
            hidden_activation=Sigmoid(), 
            input_activation=Sigmoid(), 
            output_activation=Sigmoid()) -> object:
        """
        :X: ndarray with N stimulus by n features.
        :d: 1darray or ndarray with N samples (one per stimulus) and m outputs.
        :n_hiddenLayers: positive integer that represents the number of hidden layers.
        :n_hiddenNeurons: list with positive integers, representing
        the number of neurons per hidden layer.
        :hidden_activation: list of activation functions. Must have the same length of n_hiddenNeurons,
        unless it is the same activation function for all layers. 
        """

        ## Number of perceptrons in the hidden layer.
        # Make a n_hiddenNeurons a list of a proper length.
        if not isinstance(n_hiddenNeurons, list):
            n_hiddenNeurons = [n_hiddenNeurons]

        if n_hiddenLayers==len(n_hiddenNeurons):
            self.n_hiddenNeurons = n_hiddenNeurons 
        elif len(n_hiddenNeurons)==1:
            self.n_hiddenNeurons = n_hiddenNeurons*n_hiddenLayers
        else:
            raise Exception('n_hiddenNeurons must be a list of positive integers of length\
                             n_hiddenLayers or a positive integer.')

        ## Activation functions.
        # Make a hidden_activation a list of a proper length.
        if not isinstance(hidden_activation, list):
            hidden_activation = [hidden_activation]

        if n_hiddenLayers==len(hidden_activation):
            hidden_activation = hidden_activation
        elif len(hidden_activation)==1:
            hidden_activation = hidden_activation*n_hiddenLayers
        else:
            raise Exception('hidden_activation must be a list of functions of length n_hiddenLayers or a function.')
        
        self.activation_functions = [input_activation] + hidden_activation + [output_activation]
        # Number of hidden layers.
        self.n_hiddenLayers = n_hiddenLayers

        
        # Dimensions.
        self.n_stimulus, self.n_features = X.shape[0:2]
        try:
           self.m_outputs = d.shape[1]
        except IndexError:
           self.m_outputs = 1 
        
        # Auxiliar list with the dimensions of the weights matrixes.
        # [n, n, l1, l2,..., lL, m]
        n_neurons = 2*[self.n_features] + self.n_hiddenNeurons + [self.m_outputs]

        # Initialize the neural networks.
        network = []
        for neuron in range(1, len(n_neurons)):
          # Make a dense layer.
          network.append(
              Dense(n_neurons[neuron-1], n_neurons[neuron])
              )
          network.append(self.activation_functions[neuron-1])

        self.network = network

    def predict(self, input):
        # The first layer receives the input data.
        output = input
        # Forward pass.
        for layer in self.network:
            output = layer.forward(output)
        return output
    
    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray, 
              loss=instantEnergy,
              loss_prime=instantEnergy_prime,
              n_epochs: int=1000, 
              learning_rate: float=1,
              momentum: float=0,
              verbose: bool=True):
        
        start = time.time()
        n_samples = len(x_train)
        # Preallocate an array to store the errors.
        errors = np.empty(n_epochs)
        # For each epoch...
        for epoch in range(n_epochs):
            # Initialize the error for this epoch.
            error = 0
            # For each training sample...
            for x, y in zip(x_train, y_train):
                # Forward pass.
                output = self.predict(x)

                # Update the error for this epoch.
                error = error + loss(y, output)

                # Backward pass.
                grad = loss_prime(y, output)
                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate, momentum)

            # Store the error for this epoch as the average error for all training samples.
            errors[epoch] = error / n_samples
            # If verbose, print the error for this epoch.
            if verbose:
                if (epoch+1) % 100 == 0:
                    print(f"{epoch + 1}/{n_epochs}:\n\tError={error}.\n\tElapsed time: {time.time() - start}")

        print(f'Total training time: {time.time() - start}')
        print(f'Final error: {errors[-1]}')
        self.errors = errors

    def get_weights(self):
        weights = []
        for layer in self.network:
            if isinstance(layer, Dense):
                weights.append(layer.weights)
        return weights
    
    def plot_predictions(self, x_test, y_test, verbose=True):
        # If there are more than 1 output, plot only the first one.
        if self.m_outputs > 1:
            y_test = y_test[:, 0]

        y_test = np.squeeze(y_test)
        y_pred = np.zeros(y_test.shape)
        for i, x in zip(range(len(y_test)), x_test):
            y_pred[i] = self.predict(x)
            if verbose:
                print(f'Expected: {y_test[i]}, actual: {y_pred[i]}')
        
        x_test = np.squeeze(x_test)
        # Plot subplots of each feature and the prediction vs the true value.
        # If there are more than 3 features, plot only the first 3.
        if self.n_features > 3:
            x_test = x_test[:, :3]
            n_features = 3
        else:
            n_features = self.n_features

        fig, axs = plt.subplots(1, n_features, figsize=(10, 10))
        for i in range(n_features):
            axs[i].grid()
            axs[i].plot(x_test[:, i], y_test, 'o', label='True')
            axs[i].plot(x_test[:, i], y_pred, 'x', label='Predicted')
            axs[i].legend()
            axs[i].set_xlabel(f'Feature {i}')
            axs[i].set_ylabel('Output')
            axs[i].set_title(f'Feature {i} vs Output')
        plt.show()

        # PLot the prediction vs the true value.
        fig = plt.figure(figsize=(10, 10))
        plt.plot(y_test, y_pred, 'o')
        plt.grid()
        plt.xlabel('Expected')
        plt.ylabel('Predicted')
        lb = np.min(np.vstack([y_test, y_pred]))
        ub = np.max(np.vstack([y_test, y_pred]))
        plt.ylim(lb, ub)
        plt.xlim(lb, ub)
        plt.title('Predicted vs Expected')
        plt.show()

    def plot_errors(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.errors)
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.title('Error vs Epoch')
        plt.show()