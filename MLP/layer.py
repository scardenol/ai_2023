class Layer:
    # Parent class for all layers.
    # It defines the interface that all layers must implement.
    def __init__(self):
        # TODO: initialize the layer.
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: compute the forward pass and return it.
        pass

    def backward(self, output_gradient, learning_rate, momentum):
        # TODO: compute the backward pass and return it.
        pass