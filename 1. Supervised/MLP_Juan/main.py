import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dense import Dense
from activations import Tanh, Sigmoid, ReLU, Ramp
from network import NeuralNetwork

# Generate the training data.
type = 'xor'
if type == 'xor':
    X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
    Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))
elif type == 'decoder':
    X = np.random.uniform(0.2, 0.8, size=(600, 20, 1))
    Y = X
else:
    X = np.random.uniform(0.2, 0.8, size=(600, 20, 1))
    Y = np.mean(np.power(X, 2), axis=1).reshape(600, 1, 1)
          # + np.random.uniform(0.2, 0.3, size=(600, 1, 1))

# Create the network.
nn = NeuralNetwork(X, Y, 1, 3, Tanh(), Tanh(), Tanh())

# Train the network.
nn.train(X, Y, n_epochs=1000, learning_rate=0.1, momentum=0.0)

# Plot the errors and predictions.
nn.plot_errors()
nn.plot_predictions(X, Y)

# Print the predictions for the training data.
if type == 'xor':

    # decision boundary plot
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            z = nn.predict([[x], [y]])
            points.append([x, y, z[0,0]])

    points = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
    plt.show()
