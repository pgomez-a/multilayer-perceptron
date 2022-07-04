from Perceptron import Perceptron
import numpy as np

class LayerPerceptron(object):
    """
    Simple layer perceptron architecture of a multilayer-perceptron.
    """
    def __init__(self, weights):
        """
        Initializes a layer of a multilayer-perceptron model.
        """
        if not type(weights) == np.ndarray or weights.size == 0 or weights.ndim != 2:
            raise Exception("weights must be a non-empty numpy ndarray of dimension 2.")
        self.neurons = weights.shape[0]
        self.dendrites = weights.shape[1]
        self.perceptrons = np.zeros(self.neurons, dtype = object)
        for pos in range(self.neurons):
            self.perceptrons[pos] = Perceptron(weights[pos])
        return

    def __str__(self):
        """
        Describes the structure of the layer.
        """
        return "Layer Perceptron has {} neurons with {} dendrites.".format(self.neurons, self.dendrites)

    def feed_forward(self, inputs):
        """
        Performs a feed forward operation with the stored weights.
        """
        if inputs.ndim != 1 or inputs.size != self.dendrites:
            raise Exception("inputs argument must be a numpy ndarray of dimension 1 with size {}.".format(self.dendrites))
        output = np.zeros(self.neurons + 1)
        output[0] = 1.0
        for pos in range(self.neurons):
            transfer = self.perceptrons[pos].transfer(inputs)
            activation = self.perceptrons[pos].activation(transfer)
            output[pos + 1] = activation
        return output
