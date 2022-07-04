from LayerPerceptron import LayerPerceptron
from Perceptron import Perceptron
import numpy as np

class HiddenLayerPerceptron(LayerPerceptron):
    """
    Hidden layer perceptron architecture of a multilayer-perceptron.
    """
    def __init__(self, weights):
        """
        Initializes a hidden layer of a multilayer-perceptron model.
        """
        super().__init__(weights)
        return

    def __str__(self):
        """
        Describes the structure of the hidden layer.
        """
        return "Hidden Layer Perceptron has {} neurons with {} dendrites.".format(self.size, self.dendrites)

    def feed_forward(self, inputs):
        """
        Performs a feed forward operation with the stored weights.
        """
        if inputs.ndim != 1 or inputs.size != self.dendrites:
            raise Exception("inputs argument must be a numpy ndarray of dimension 1 with size {}.".format(self.dendrites))
        output = np.zeros(self.size + 1)
        output[0] = 1.0
        for pos in range(self.size):
            transfer = self.perceptrons[pos].transfer(inputs)
            activation = self.perceptrons[pos].activation(transfer)
            output[pos + 1] = activation
        return output
