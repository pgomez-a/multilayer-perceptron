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

    def get_weights(self):
        """
        Returns the weights of the layers of the multilayer-perceptron model.
        """
        weights = list()
        for layer in range(self.__size):
            weights.append(self.__layers[layer].get_weights())
        return weights

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

    def forward_propagation(self, inputs):
        """
        Performs forward propagation with the stored weights.
        """
        for layer_pos in range(self.__size):
            if layer_pos == 0:
                for input_pos in range(inputs.size):
                    inputs[input_pos] = self.__layers[layer_pos].feed_forward(np.full((1), inputs[input_pos]))[1]
                inputs = np.insert(inputs, 0, 1)
            else:
                inputs = self.__layers[layer_pos].feed_forward(inputs)
        return self.__softmax(inputs[1:])

    def __create_gradient_record(self):
        """
        Creates the gradient record needed to perform back propagation.
        """
        gradient_record = list()
        weights = self.get_weights()[1:]
        for row in range(len(weights)):
            gradient_record.append(np.zeros(weights[row].shape))
        return gradient_record

    def __get_activation_values(self, inputs):
        """
        Performs forward propagation for each of the layers of the model.
        """
        activation_values = list()
        for layer_pos in range(self.__size):
            if layer_pos == 0:
                for input_pos in range(inputs.size):
                    inputs[input_pos] = self.__layers[layer_pos].feed_forward(np.full((1), inputs[input_pos]))[1]
                inputs = np.insert(inputs, 0, 1)
            elif layer_pos == self.__size - 1:
                inputs = self.__layers[layer_pos].feed_forward(inputs)[1:]
            else:
                inputs = self.__layers[layer_pos].feed_forward(inputs)
            activation_values.append(inputs)
        return activation_values

    def __get_error_values(self, activation_values, Y):
        """
        Computes the error value for each of the neurons of the model.
        """
        error_values = list()
        for layer_pos in range(self.__size, 1, -1):
            if layer_pos == self.__size:
                error = activation_values[layer_pos - 1] - Y
                error_values.append(error)
            else:
                if layer_pos == self.__size - 1:
                    error = np.matmul(self.__layers[layer_pos].get_weights().transpose(), error)
                else:
                    error = np.matmul(self.__layers[layer_pos].get_weights().transpose(), error[1:])
                error *= activation_values[layer_pos - 1]
                error *= (1 - activation_values[layer_pos - 1])
                error_values.append(error)
        error_values.append(np.zeros(activation_values[0].size))
        error_values.reverse()
        return error_values

    def back_propagation(self, X, Y, alpha = 0.1):
        """
        Performs back propagation with the stored weights.
        """
        gradient_record = self.__create_gradient_record()
        for row in range(X.shape[0]):
            activation_values = self.__get_activation_values(X[row])
            error_values = self.__get_error_values(activation_values, Y[row])
            gradient = list()
            for pos in range(len(activation_values) - 1):
                cost = np.matmul(error_values[pos + 1].reshape((-1, 1)), activation_values[pos].reshape((-1, 1)).transpose())
                if pos != self.__size - 2:
                    cost = cost[1:]
                gradient.append(cost)
            for pos in range(len(gradient_record)):
                gradient_record[pos] += gradient[pos]
        weights = self.get_weights()
        for pos in range(len(gradient_record)):
            gradient_record[pos] /= X.shape[0]
            weights[pos + 1] -= alpha * gradient_record[pos]
            self.__layers[pos + 1].set_weights(weights[pos + 1])
        return
