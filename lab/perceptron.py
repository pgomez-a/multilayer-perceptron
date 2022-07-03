import numpy as np

class Perceptron(object):
    """
    Simple model of a biological neuron in an artificial neural network. 
    """
    def __init__(self, weights):
        """
        Initializes a perceptron by giving its corresponding weights.
        """
        if not type(weights) == np.ndarray or weights.ndim != 1 or weights.size == 0:
            raise Exception("weights must be a non-empty numpy ndarray of dimension 1.")
        self.weights = weights
        self.dendrites = weights.size
        return

    def __str__(self):
        """
        Describes the structure of the perceptron.
        """
        return "Perceptron has {} dendrites with weights: {}".format(self.dendrites, self.weights)

    def get_weights(self):
        """
        Returns the weights used to compute the tranfer and activation functions.
        """
        return self.weights

    def set_weights(self, weights):
        """
        Sets the weights used to compute the tranfer and activation functions.
        """
        if not type(weights) == np.ndarray or weights.ndim != 1 or weights.size == 0:
            raise Exception("weights must be a non-empty numpy ndarray of dimension 1.")
        self.weights = weights
        self.dendrites = weights.size
        return

    def transfer(self, inputs):
        """
        Computes multiple inputs so that the activation function can be applied.
        """
        if not type(inputs) == np.ndarray or inputs.ndim != 1 or inputs.size == 0:
            raise Exception("inputs must be a non-empty numpy ndarray of dimension 1.")
        if inputs.size != self.weights.size:
            raise Exception("inputs({}) must be the same size as weights({}).".format(inputs.size, self.weights.size))
        return np.dot(inputs, self.weights)

    def linear_act(self, value):
        """
        Calculates a linear activation function.
        """
        if not type(value) == int and not type(value) == float:
            raise Exception("value argument must be a numerical value.")
        return float(value)

    def sigmoid_act(self, value):
        """
        Calculates a sigmoid activation function.
        """
        if not type(value) == int and not type(value) == float:
            raise Exception("value argument must be a numerical value.")
        return 1 / (1 + np.exp(-value))

    def tanh_act(self, value):
        """
        Calculates a tanh activation function.
        """
        if not type(value) == int and not type(value) == float:
            raise Exception("value argument must be a numerical value.")
        return (2 / (1 + np.exp(-2 * value))) - 1

    def relu_act(self, value):
        """
        Calculates a ReLU activation function.
        """
        if not type(value) == int and not type(value) == float:
            raise Exception("value argument must be a numerical value.")
        if value <= 0:
            return 0.0
        return float(value)

    def activation(self, value, func_name = "sigmoid"):
        """
        Computes the activation value with the specified activation function.
        """
        if func_name == "linear":
            return self.linear_act(value)
        if func_name == "sigmoid":
            return self.sigmoid_act(value)
        if func_name == "tanh":
            return self.tanh_act(value)
        if func_name == "relu":
            return self.relu_act(value)
        raise Exception("{} activation function is not defined.".format(func_name))
