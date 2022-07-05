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

if __name__ == '__main__':
    dataset = read_dataset()
    weights = get_weights()
    multilayer = MultilayerPerceptron(dataset.shape[1] - 1, weights)
    sys.exit(0)
