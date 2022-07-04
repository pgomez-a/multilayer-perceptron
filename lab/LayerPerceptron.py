from Perceptron import Perceptron
import numpy as np

class LayerPerceptron(object):
    """
    Simple layer perceptron architecture of a multilayer-perceptron.
    """
    def __init__(self, size, dendrites, weights):
        """
        Initializes a layer of a multilayer-perceptron model.
        """
        if not type(size) == int or size <= 0:
            raise Exception("size argument must be a positive integer value.")
        if not type(dendrites) == int or dendrites <= 0:
            raise Exception("dendrites argument must be a positive integer value.")
        if not type(weights) == np.ndarray or weights.size == 0:
            raise Exception("weights must be a non-empty numpy ndarray.")
        if weights.ndim != 2 or size != weights.shape[0] or dendrites != weights.shape[1]:
            raise Exception("weights{} must be of dimension ({},{}).".format(weights.shape, size, dendrites))
        self.size = size
        self.dendrites = dendrites
        self.perceptrons = np.zeros(size, dtype = object)
        for pos in range(size):
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
