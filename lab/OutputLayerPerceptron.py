from LayerPerceptron import LayerPerceptron
from Perceptron import Perceptron
import numpy as np

class OutputLayerPerceptron(LayerPerceptron):
    """
    Output layer perceptron architecture of a multilayer-perceptron.
    """
    def __init__(self, size, dendrites, weights):
        """
        Initializes an output layer of a multilayer-perceptron model.
        """
        super().__init__(size, dendrites, weights)
        return

    def __str__(self):
        """
        Describes the structure of the output layer.
        """
        return "Output Layer Perceptron has {} neurons with {} dendrites.".format(self.size, self.dendrites)

    def feed_forward(self, inputs):
        """
        Performs a feed forward operation with the stored weights.
        """
        if inputs.ndim != 1 or inputs.size != self.dendrites:
            raise Exception("inputs argument must be a numpy ndarray of dimension 1 with size {}.".format(self.dendrites))
        output = np.zeros(self.size)
        for pos in range(self.size):
            transfer = self.perceptrons[pos].transfer(inputs)
            activation = self.perceptrons[pos].activation(transfer)
            output[pos] = activation
        return output
