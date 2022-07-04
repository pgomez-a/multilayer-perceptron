from LayerPerceptron import LayerPerceptron
from Perceptron import Perceptron
import numpy as np

class InputLayerPerceptron(LayerPerceptron):
    """
    Input layer perceptron architecture of a multilayer-perceptron.
    """
    def __init__(self, size):
        """
        Initializes an input layer of a multilayer-perceptron model.
        """
        if not type(size) == int or size <= 0:
            raise Exception("size argument must be a positive integer value.")
        super().__init__(np.ones((size, 1)))
        return

    def __str__(self):
        """
        Describes the structure of the input layer.
        """
        return "Input Layer Perceptron has {} neurons with {} dendrites.".format(self.size, self.dendrites)

    def feed_forward(self, inputs):
        """
        Performs a feed forward operation with the stored weights.
        """
        if inputs.ndim != 1 or inputs.size != self.size:
            raise Exception("inputs argument must be a numpy ndarray of dimension 1 with size {}.".format(self.size))
        output = np.zeros(self.size + 1)
        output[0] = 1.0
        for pos in range(self.size):
            transfer = self.perceptrons[pos].transfer(np.array([inputs[pos]]))
            activation = self.perceptrons[pos].activation(transfer)
            output[pos + 1] = activation
        return output
