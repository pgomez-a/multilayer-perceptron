from LayerPerceptron import LayerPerceptron
import numpy as np

class MultilayerPerceptron(object):
    """
    Multilayer perceptron architecture.
    """
    def __init__(self, input_len, weights):
        """
        Initializes a multilayer-perceptron model.
        """
        if not type(input_len) == int or input_len <= 0:
            raise Exception("input_len must be a numerical positive value.")
        if not type(weights) == list or len(weights) == 0:
            raise Exception("weights must be a list of non-empty numpy ndarray of dimension 2.")
        for elem in weights:
            if not type(elem) == np.ndarray or elem.size == 0 or elem.ndim != 2:
                raise Exception("weights must be a non-empty numpy ndarray of dimension 3.")
        self.__size = 1 + len(weights)
        self.__layers = np.zeros(self.__size, dtype = object)
        for pos in range(self.__size):
            if pos == 0:
                self.__layers[pos] = LayerPerceptron(np.ones((input_len, 1)))
            else:
                self.__layers[pos] = LayerPerceptron(weights[pos - 1])
                if self.__layers[pos].get_dendrites() != self.__layers[pos - 1].get_neurons() + 1:
                    raise Exception("The dendrites of a neuron must be the same as the number of neurons of the previous layer.")
        return

    def __str__(self):
        """
        Describes the structure of the multilayer perceptron.
        """
        msg = "Multilayer Perceptron has {} layers where:\n".format(self.__size)
        pos = 0
        for layer in self.__layers:
            if pos == 0:
                msg += "\tInput Layer Perceptron has {} neurons where:\n".format(layer.get_neurons())
                pos += 1
            else:
                msg += "\tHidden Layer Perceptron has {} neurons where:\n".format(layer.get_neurons())
            for neuron in layer.get_perceptrons():
                msg += "\t\tPerceptron has {} dendrites with weights: {}\n".format(neuron.get_dendrites(), neuron.get_weights())
        return msg

    @staticmethod
    def __softmax(vector):
        """
        Applies the softmax function to the given vector.
        """
        denominator = 0
        for scalar in vector:
            denominator += np.exp(scalar)
        for pos in range(vector.size):
            vector[pos] = np.exp(vector[pos]) / denominator
        return vector

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
        for layer_pos in range(self.__size):
            if layer_pos == 0:
                for input_pos in range(inputs.size):
                    inputs[input_pos] = self.__layers[layer_pos].feed_forward(np.full((1), inputs[input_pos]))[1]
                inputs = np.insert(inputs, 0, 1)
            else:
                inputs = self.__layers[layer_pos].feed_forward(inputs)
        return self.__softmax(inputs[1:])
