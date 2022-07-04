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
        self.size = weights.shape[0]
        self.dendrites = weights.shape[1]
        self.perceptrons = np.zeros(self.size, dtype = object)
        for pos in range(self.size):
            self.perceptrons[pos] = Perceptron(weights[pos])
        return

    def __str__(self):
        """
        Describes the structure of the layer.
        """
        return "Layer Perceptron has {} neurons with {} dendrites.".format(self.size, self.dendrites)

    def feed_forward(self, inputs):
        """
        Performs a feed forward operation with the stored weights.
        """
        print("This is a general Layer Perceptron Arhitecture.")
        print("You have to determine the type of the layer if you want to perform feed forward.")
        return
