from LayerPerceptron import LayerPerceptron
import numpy as np

class MultilayerPerceptron(object):
    """
    Multilayer perceptron architecture.
    """
    def __init__(self, weights):
        """
        Initializes a multilayer-perceptron model.
        """
        if not type(weights) == list or len(weights) == 0:
            raise Exception("weights must be a list of non-empty numpy ndarray of dimension 2.")
        for elem in weights:
            if not type(elem) == np.ndarray or elem.size == 0 or elem.ndim != 2:
                raise Exception("weights must be a non-empty numpy ndarray of dimension 3.")
        self.__size = len(weights)
        self.__layers = np.zeros(self.__size, dtype = object)
        for pos in range(self.__size):
            self.__layers[pos] = LayerPerceptron(weights[pos])
            if pos != 0 and self.__layers[pos].get_dendrites() != self.__layers[pos - 1].get_neurons() + 1:
                raise Exception("The dendrites of a neuron must be the same as the number of neurons of the previous layer.")
        return

    def __str__(self):
        """
        Describes the structure of the multilayer perceptron.
        """
        msg = "Multilayer Perceptron has {} layers where:\n".format(self.__size)
        for layer in self.__layers:
            msg += "\tLayer Perceptron has {} neurons where:\n".format(layer.get_neurons())
            for neuron in layer.get_perceptrons():
                msg += "\t\tPerceptron has {} dendrites with weights: {}\n".format(neuron.get_dendrites(), neuron.get_weights())
        return msg

    def get_size(self):
        """
        Returns the number of layers of the multilayer-perceptron model.
        """
        return self.__size

    def get_layers(self):
        """
        Returns the layers of the multilayer-perceptron model.
        """
        return self.__layers

    def feed_forward(self, inputs):
        """
        Performs a feed forward operation with the stored weights.
        """
        inputs = np.insert(inputs, 0, 1)
        for layer in self.__layers:
            inputs = layer.feed_forward(inputs)
        return inputs[1:]
