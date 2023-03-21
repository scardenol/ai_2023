# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create the mlp class
class mlp:
    """This is the Multi Layer Perceptron class. An object of this class is an instance of a neural network
    with a shallow or a deep architecture depending on the number of hidden layers (depth) and neurons per
    hidden layer (base).
    """
    def __init__(
        self,
        inputs,
        outputs,
        validation_inputs,
        validation_outputs,
        activation=None,
        hidden_layer=None,
        learning_rate=1,
        epochs=100,
        tol=1e-2,
    ) -> None:  # sourcery skip: dict-assign-update-to-union, simplify-dictionary-update
        if activation is None:
            activation = {"input": "sigmoid", "hidden": "sigmoid", "output": "sigmoid"}
        if hidden_layer is None:
            hidden_layer = {"layers": 1, "neurons": 3}
        hidden_layer["activation"] = activation["hidden"]
        self.X = np.array(inputs)
        self.Y = np.array(outputs)
        self.validation = {
            "X_val": np.array(validation_inputs),
            "Y_val": np.array(validation_outputs),
        }
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        input_layer = {"neurons": np.ndim(inputs), "activation": activation["input"]}
        self.layers = {"input_layer": input_layer}
        for layer in range(hidden_layer["layers"]):
            layer_name = f"hidden_layer_{str(layer)}"
            self.layers[layer_name] = {
                "neurons": hidden_layer["neurons"],
                "activation": activation["hidden"],
            }
        output_layer = {"neurons": np.ndim(outputs), "activation": activation["output"]}
        self.layers.update({"output_layer": output_layer})
        layers_names = list(self.layers.keys())
        self.layers_names = layers_names
        for i in range(len(layers_names)):
            layer = layers_names[i]
            if i == 0:
                self.layers[layer]["weights"] = np.random.uniform(
                    low=-1,
                    high=1,
                    size=(self.layers[layer]["neurons"], self.layers[layer]["neurons"]),
                )
            else:
                previous = layers_names[i - 1]
                self.layers[layer]["weights"] = np.random.uniform(
                    low=-1,
                    high=1,
                    size=(
                        self.layers[previous]["neurons"],
                        self.layers[layer]["neurons"],
                    ),
                )
        self.validation["layers"] = self.layers

    def phi(self, x, activation):
        """This is the activation function applied to all the neurons of a specific layer.

        Args:
            x (np.array): a number, vector or array of numbers.
            activation (str): a string with the name of the desired activation function.

        Returns:
            np.array: the resulting transformation of the input data object x.
        """
        if activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "relu":
            return np.maximum(0, x)

    def dphi(self, x, activation):
        """This is the derivate of the activation function applied to all the neurons of a specific layer.

        Args:
            x (np.array): a number, vector or array of numbers.
            activation (str): a string with the name of the desired activation function.

        Returns:
            np.array: the resulting transformation of the input data object x.
        """
        if activation == "sigmoid":
            return x * (1 - x)
        elif activation == "tanh":
            return 1 - np.tanh(x) ** 2
        elif activation == "relu":
            return (x > 0) * 1

    def e(self, d, y):
        """The error function, defined as the difference between the desired (or expected) and real d - y.

        Args:
            d (np.array): the desired vector.
            y (np.array): the real vector.

        Returns:
            np.array: the error vector.
        """
        if np.shape(d) != np.shape(y):
            d = np.reshape(d, newshape=y.shape)
        return d - y

    def E(self, e):
        """The instantaneous energy error, it measures the lack of effectiveness for each neuron in a
        specific layer.

        Args:
            e (np.array): the error vector.

        Returns:
            np.array: the instantaneous energy vector.
        """
        return 0.5 * (sum(e**2))

    def E_av(self, E):
        """The average instantaneous energy error, it measures the complete network error which makes it the
        loss function. The goal is to find the weights that minimize this average error of estimation.

        Args:
            E (np.array): the instantaneous energy vector.

        Returns:
            float: the complete network error of estimation.
        """
        return np.mean(E)

    def forward_pass(self):
        """Calculates the output for each layer. This is the first step and the results are needed for
        the backward pass.
        """
        for i in range(len(self.layers_names)):
            layer = self.layers_names[i]
            if i == 0:  # input layer
                self.layers[layer]["local_field"] = np.dot(
                    self.X, self.layers[layer]["weights"]
                )
            else:  # other layers
                previous = self.layers_names[i - 1]
                self.layers[layer]["local_field"] = np.dot(
                    self.layers[previous]["output"], self.layers[layer]["weights"]
                )
            self.layers[layer]["output"] = self.phi(
                self.layers[layer]["local_field"], self.layers[layer]["activation"]
            )

    def backward_pass(self):
        """This method is the backpropagation without updating the weights, hence its name
        backward pass. It calculates the local gradients in reverse from the output to the input layer.
        """
        # Backpropagation
        for i in reversed(range(len(self.layers_names))):
            layer = self.layers_names[i]
            if i == len(self.layers_names) - 1:  # output layer
                error = self.e(self.Y, self.layers[layer]["output"])
                energy_error = self.E(error)
                self.network_error = np.append(
                    self.network_error, self.E_av(energy_error)
                )
                self.layers[layer]["error"] = error
            else:  # other layers
                previous = self.layers_names[i + 1]
                error = self.layers[previous]["local_gradient"].dot(
                    self.layers[previous]["weights"].T
                )
            local_gradient = error * self.dphi(
                self.layers[layer]["output"], self.layers[layer]["activation"]
            )
            self.layers[layer]["local_gradient"] = local_gradient
        self.delta_k = np.append(
            self.delta_k, np.mean(self.layers["output_layer"]["local_gradient"])
        )

    def update_weights(self):
        """This method is the last part of backpropagation, where the classical gradient
        descent algorithm its used to update the weights.
        """
        # update weights using gradient descent
        for i in range(len(self.layers_names)):
            layer = self.layers_names[i]
            w = self.layers[layer]["weights"]
            if i == 0:  # input layer
                dw = self.X.T.dot(self.layers[layer]["local_gradient"])
            else:
                previous = self.layers_names[i - 1]
                dw = self.layers[previous]["output"].T.dot(
                    self.layers[layer]["local_gradient"]
                )
            w += self.learning_rate * dw
            self.layers[layer]["weights"] = w

    def validate(self):
        """This is the validation method. Here the validation set its used to perform
        a forward pass of the network, then a backward pass to the output layer
        calculating only the output local gradient and the network error.
        This method doesn't perform weights updates and does not interfere with the training
        of the network. 
        """

        # Forward pass with validation input
        for i in range(len(self.layers_names)):
            layer = self.layers_names[i]
            if i == 0:  # input layer
                self.validation["layers"][layer]["local_field"] = np.dot(
                    self.validation["X_val"],
                    self.validation["layers"][layer]["weights"],
                )
            else:  # other layers
                previous = self.layers_names[i - 1]
                self.validation["layers"][layer]["local_field"] = np.dot(
                    self.validation["layers"][previous]["output"],
                    self.validation["layers"][layer]["weights"],
                )
            self.validation["layers"][layer]["output"] = self.phi(
                self.validation["layers"][layer]["local_field"],
                self.validation["layers"][layer]["activation"],
            )

        # Backward pass only to calculate the network error and output gradient with validation set
        layer = "output_layer"
        error = self.e(
            self.validation["Y_val"], self.validation["layers"][layer]["output"]
        )
        energy_error = self.E(error)
        self.validation["network_error"] = np.append(
            self.validation["network_error"], self.E_av(energy_error)
        )
        self.validation["layers"][layer]["error"] = error
        local_gradient = error * self.dphi(
            self.validation["layers"][layer]["output"],
            self.validation["layers"][layer]["activation"],
        )
        self.validation["layers"][layer]["local_gradient"] = local_gradient
        self.validation["delta_k"] = np.append(
            self.validation["delta_k"],
            np.mean(self.validation["layers"][layer]["local_gradient"]),
        )

    def learn(self, print_progress=False):
        """This is the main method. In here the learning process its performed on the
        training set and evaluated on the validation set. The main desired results are collected here.

        Args:
            print_progress (bool, optional): An optional parameter to flag
            whether the user wants to print the progress of the network learning
            or not. Defaults to False.
        """
        self.network_error = []
        self.delta_k = []
        self.validation["network_error"] = []
        self.validation["delta_k"] = []
        for epoch in range(self.epochs):
            self.forward_pass()
            self.backward_pass()
            if self.network_error[-1] < self.tol:
                break
            self.update_weights()
            self.validate()
            if print_progress:
                print("epoch =", epoch, "error =", self.network_error[-1])
        if print_progress:
            print(
                "desired:\n", self.Y, "real:\n", self.layers["output_layer"]["output"]
            )

    def predict(self, X_test, Y_test):
        """The method to make predictions. This method is created to make
        a forward pass of the trained and validated network and predict the output
        based on a test set.

        Args:
            X_test (array): input features of the test set.
            Y_test (array): outputs of the test set.
        """
        self.test = {"X_test": np.array(X_test), "Y_test": np.array(Y_test)}
        self.test["layers"] = self.layers

        # Make a forward pass to calculate the network output with the current data
        for i in range(len(self.layers_names)):
            layer = self.layers_names[i]
            if i == 0:  # input layer
                self.test["layers"][layer]["local_field"] = np.dot(
                    self.test["X_test"],
                    self.test["layers"][layer]["weights"],
                )
            else:  # other layers
                previous = self.layers_names[i - 1]
                self.test["layers"][layer]["local_field"] = np.dot(
                    self.test["layers"][previous]["output"],
                    self.test["layers"][layer]["weights"],
                )
            self.test["layers"][layer]["output"] = self.phi(
                self.test["layers"][layer]["local_field"],
                self.test["layers"][layer]["activation"],
            )
        self.test["Y_hat"] = self.test["layers"]["output_layer"]["output"]


# Read the data and sample the desired amount of observations
data = pd.read_csv("datosIA.txt", header=None)
X = data.loc[:499, [0, 1]].values
Y = data.loc[:499, 2].values

# Normalize both the input features and the output data to [0,1]
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))

# Define a split data function for the train-validation-test split
def split_data(data, train_ratio=0.6, validation_ratio=0.2):
    train_index = int(len(data) * train_ratio)
    validation_index = train_index + int(len(data) * validation_ratio)
    return (
        data[:train_index],
        data[train_index:validation_index],
        data[validation_index:],
    )


# Split data into train (60%), validation (20%) y test (20%)
X_train, X_validation, X_test = split_data(X, train_ratio=0.6, validation_ratio=0.2)
Y_train, Y_validation, Y_test = split_data(Y, train_ratio=0.6, validation_ratio=0.2)


# Do the experiment for each model and store the results

np.random.seed(314159265358979323846264338)

# General parameters
L = 3  # Max number of hidden layers
li = 5  # Max number of neurons per hidden layer
learning_rate = [0.2, 0.5, 0.9]  # Learning rates

# Variables to collect and store results
networks_errors = {}
networks_errors_val = {}
parameters = {}
deltas_k = {}
deltas_k_val = {}
y_hats = {}
models = {}

# Then main experiment cycle: iterate over base, learning rate and depth
for L in range(1, L + 1):
    for lr in learning_rate:
        for li in range(1, li + 1):
            p = mlp(
                inputs=X_train,
                outputs=Y_train,
                validation_inputs=X_validation,
                validation_outputs=Y_validation,
                epochs=50,
                hidden_layer={"layers": L, "neurons": li},
                learning_rate=lr,
                tol=1e-2,
            )
            p.learn(print_progress=False)
            model = f"L = {L}, li = {li}, lr = {lr}"
            networks_errors[model] = p.network_error
            networks_errors_val[model] = p.validation["network_error"]
            parameters[model] = np.ndim(X_train) * L * li * np.ndim(Y_train)
            deltas_k[model] = p.delta_k
            deltas_k_val[model] = p.validation["delta_k"]
            p.predict(X_test, Y_test)
            y_hats[model] = p.test["Y_hat"]
    models[f"layer_{L}"] = {
        "networks_errors": networks_errors,
        "networks_errors_val": networks_errors_val,
        "parameters": parameters,
        "deltas_k": deltas_k,
        "deltas_k_val": deltas_k_val,
        "y_hats": y_hats,
    }
    models[f"layer_{L}"]["names"] = list(models[f"layer_{L}"]["networks_errors"].keys())
    networks_errors = {}
    networks_errors_val = {}
    parameters = {}
    deltas_k = {}
    deltas_k_val = {}

# Manage the results, apply the selection criterion, plot and save

# Create the Figs folder to store the plots or skip if it exists
os.makedirs(
    "Figs", exist_ok=True
) 

# For each layer use the results, select the best, worst and median model and plot all the results
for layer in models:
    networks_errors = models[layer]["networks_errors"]
    parameters = models[layer]["parameters"]
    deltas_k = models[layer]["deltas_k"]
    networks_errors_val = models[layer]["networks_errors_val"]
    deltas_k_val = models[layer]["deltas_k_val"]

    # Sort by last network_error (lowest-largest) in validation set
    sort1 = dict(sorted(networks_errors_val.items(), key=lambda x: x[1][-1]))
    # Sort by number of parameters (lowest-largest)
    sort2 = dict(sorted(sort1.items(), key=lambda x: parameters[x[0]]))
    sort2_list = list(sort2.items())
    best = sort2_list[0][0]
    worst = sort2_list[-1][0]
    median = sort2_list[len(sort2_list) // 2 - 1][0]

    # Plot training results
    my_dpi = 96  # My screen DPI
    my_width = 1366  # Desired width resolution
    my_length = 655  # Desired length resolution

    title = f"Training for ANN with L = {layer[-1]}"
    fig, ax = plt.subplots(
        1, 2, num=title, figsize=(my_width / my_dpi, my_length / my_dpi), dpi=my_dpi
    )
    fig.suptitle(title)
    ax[0].plot(networks_errors[best], ".-", label=f"best: {best}")
    ax[0].plot(networks_errors[worst], ".-", label=f"worst: {worst}")
    ax[0].plot(networks_errors[median], ".-", label=f"median: {median}")
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title(r"Network error $\mathcal{E}_{av}$ over epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel(r"$\mathcal{E}_{av}$")

    ax[1].plot(deltas_k[best], ".-", label=f"best: {best}")
    ax[1].plot(deltas_k[worst], ".-", label=f"worst: {worst}")
    ax[1].plot(deltas_k[median], ".-", label=f"median: {median}")
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title(r"Average output gradient $\bar{\delta}_k$ over epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel(r"$\bar{\delta}_k$")
    plt.savefig(f'Figs/{title.replace(" ", "_")}.png')
    plt.show(block=False)

    # Plot validation results
    title = f"Validation for ANN with L = {layer[-1]}"
    fig, ax = plt.subplots(
        1, 2, num=title, figsize=(my_width / my_dpi, my_length / my_dpi), dpi=my_dpi
    )
    fig.suptitle(title)
    ax[0].plot(networks_errors_val[best], ".-", label=f"best: {best}")
    ax[0].plot(networks_errors_val[worst], ".-", label=f"worst: {worst}")
    ax[0].plot(networks_errors_val[median], ".-", label=f"median: {median}")
    ax[0].grid()
    ax[0].legend()
    ax[0].set_title(r"Network error $\mathcal{E}_{av}$ over epochs")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel(r"$\mathcal{E}_{av}$")

    ax[1].plot(deltas_k_val[best], ".-", label=f"best: {best}")
    ax[1].plot(deltas_k_val[worst], ".-", label=f"worst: {worst}")
    ax[1].plot(deltas_k_val[median], ".-", label=f"median: {median}")
    ax[1].grid()
    ax[1].legend()
    ax[1].set_title(r"Average output gradient $\bar{\delta}_k$ over epochs")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel(r"$\bar{\delta}_k$")
    plt.savefig(f'Figs/{title.replace(" ", "_")}.png')
    plt.show(block=False)

    # Plot testing results
    y_hats = models[layer]["y_hats"]
    title = f"Testing for ANN with L = {layer[-1]}"
    plt.figure(title, figsize=(my_width / my_dpi, my_length / my_dpi), dpi=my_dpi)
    plt.plot(Y_test, ".-", label="Real")
    plt.plot(y_hats[best], ".-", label=f"best: {best}")
    plt.plot(y_hats[worst], ".-", label=f"worst: {worst}")
    plt.plot(y_hats[median], ".-", label=f"median: {median}")
    plt.grid()
    plt.legend()
    plt.suptitle(title)
    plt.title("Output over time")
    plt.xlabel("Time")
    plt.ylabel("Output")
    plt.savefig(f'Figs/{title.replace(" ", "_")}.png')
    plt.show(block=False)