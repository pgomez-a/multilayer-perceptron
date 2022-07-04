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
        self.__neurons = weights.shape[0]
        self.__dendrites = weights.shape[1]
        self.__perceptrons = np.zeros(self.__neurons, dtype = object)
        for pos in range(self.__neurons):
            self.__perceptrons[pos] = Perceptron(weights[pos])
        return

    def __str__(self):
        """
        Describes the structure of the layer.
        """
        msg = "Layer Perceptron has {} neurons where:\n".format(self.__neurons)
        for neuron in self.__perceptrons:
            msg += "\tPerceptron has {} dendrites with weights: {}\n".format(neuron.get_dendrites(), neuron.get_weights())
        return msg

    def get_neurons(self):
        """
        Returns the number of neurons of the layer.
        """
        return self.__neurons

    def get_dendrites(self):
        """
        Returns the number of dendrites of the perceptron of the layer.
        """
        return self.__dendrites

    def get_perceptrons(self):
        """
        Returns the perceptrons of the layer.
        """
        return self.__perceptrons

    def feed_forward(self, inputs):
        """
        Performs a feed forward operation with the stored weights.
        """
        if inputs.ndim != 1 or inputs.size != self.__dendrites:
            raise Exception("inputs argument must be a numpy ndarray of dimension 1 with size {}.".format(self.__dendrites))
        output = np.zeros(self.__neurons + 1)
        output[0] = 1.0
        for pos in range(self.__neurons):
            transfer = self.__perceptrons[pos].transfer(inputs)
            activation = self.__perceptrons[pos].activation(transfer)
            output[pos + 1] = activation
        return output
