import pandas as pd
import numpy as np
import sys

sys.path.insert(0, './lab/')

from MultilayerPerceptron import MultilayerPerceptron

def read_dataset():
    """
    Reads the dataset passed as the first argument.
    """
    if len(sys.argv) != 3:
        print("\033[1m\033[91mError: read_dataset. evaluate_network.py takes two arguments.\n\033[0m")
        sys.exit(1)
    try:
        print("\033[1mReading dataset...\033[0m")
        dataset = pd.read_csv(sys.argv[1])
    except:
        print("\033[1m\033[91mError: read_dataset. {} can't be read.\n\033[0m".format(sys.argv[1]))
        sys.exit(1)
    return dataset

def get_weights():
    """
    Reads the weights file passed as the second argument.
    """
    try:
        with open(sys.argv[2], 'r') as f:
            print("\033[1mReading weights...\033[0m")
            weights = list()
            layers = int(f.readline())
            for layer in range(layers):
                line = f.readline().split()
                perceptrons = np.zeros((int(line[0]), int(line[1])))
                for neuron in range(perceptrons.shape[0]):
                    perceptrons[neuron] = np.array(f.readline().split())
                weights.append(perceptrons)
    except:
        print("\033[1m\033[91mError: read_dataset. {} can't be read.\n\033[0m".format(sys.argv[2]))
        sys.exit(1)
    return weights

def predict(X, multilayer, true_val, neg_val):
    """
    Performs a prediction for each of the given inputs.
    """
    print("\033[1mComputing predictions...\033[0m")
    Y_hat_num = np.zeros(X.shape[0], dtype = float)
    Y_hat_str = np.zeros(X.shape[0], dtype = object)
    for pos in range(X.shape[0]):
        val = multilayer.forward_propagation(X[pos])
        Y_hat_num[pos] = val[0] if val[0] >= val[1] else val[1]
        Y_hat_str[pos] = true_val if val[0] >= val[1] else neg_val
    return Y_hat_num, Y_hat_str

def cost(Y, Y_hat, true_val, neg_val):
    """
    Performs a binary cross-entropy error function to evaluate the accuracy of the model.
    """
    Y[Y == true_val] = 1.0
    Y[Y == neg_val] = 0.0
    error = -sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / Y.shape[0]
    print("\033[1m\nBinary cross-entropy error value is: {}.\n\033[0m".format(error))
    return

if __name__ == '__main__':
    dataset = read_dataset()
    weights = get_weights()
    multilayer = MultilayerPerceptron(dataset.shape[1] - 1, weights)
    Y = dataset.iloc[:, 0].to_numpy()
    X = dataset.iloc[:, 1:].to_numpy()
    Y_hat_num, Y_hat_str = predict(X, multilayer, true_val = 'M', neg_val = 'B')
    cost(Y, Y_hat_num, true_val = 'M', neg_val = 'B')
    sys.exit(0)
