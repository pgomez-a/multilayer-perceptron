import pandas as pd
import numpy as np
import sys

sys.path.insert(0, './lab/')

from MultilayerPerceptron import MultilayerPerceptron

def read_dataset():
    """
    Reads the dataset with the cleaned data.
    """
    if len(sys.argv) != 3:
        print("\033[1m\033[91mError: read_dataset. train_network.py takes two arguments.\n\033[0m")
        sys.exit(1)
    try:
        data_train = int(sys.argv[2])
    except:
        print("\033[1m\033[91mError: read_dataset. train_network.py second argument must be int.\n\033[0m")
        sys.exit(1)
    if data_train <= 0 or data_train >= 100:
        print("\033[1m\033[91mError: read_dataset. train_network.py second argument must be between 1-99.\n\033[0m")
        sys.exit(1)
    try:
        dataset = pd.read_csv(sys.argv[1])
        data_train = round(dataset.shape[0] * (data_train / 100))
        dataset_train = dataset.iloc[:data_train]
        dataset_test = dataset.iloc[data_train:]
    except:
        print("\033[1m\033[91mError: read_dataset. {} can't be read.\n\033[0m".format(sys.argv[1]))
        sys.exit(1)
    return dataset_train, dataset_test

def get_hidden_weights(topology, input_len):
    """
    Creates the weights that will be given to the multilayer-perceptron.
    """
    if not type(topology) == tuple:
        print("\033[1m\033[91mError: get_weights. topology must be a tuple.\n\033[0m")
        sys.exit(1)
    if not type(input_len) == int or input_len <= 0:
        print("\033[1m\033[91mError: get_weights. input_len must be a positive int.\n\033[0m")
        sys.exit(1)
    weights = list()
    for pos in range(len(topology)):
        if pos == 0:
            layer = np.random.rand(topology[pos], input_len + 1)
        else:
            layer = np.random.rand(topology[pos], topology[pos - 1] + 1)
        weights.append(layer * 10)
    return weights

def give_train_format(train, true_val, neg_val):
    """
    Formats the values that will be used to train the multilayer.
    """
    X = np.array(train.iloc[:,1:])
    Y = np.array(train.iloc[:,0])
    Y[Y == true_val] = 1.0
    Y[Y == neg_val] = 0.0
    Y_set = np.zeros((Y.shape[0], 2))
    for row in range(Y_set.shape[0]):
        if Y[row] == 1.0:
            Y_set[row][0] = 1.0
        else:
            Y_set[row][1] = 1.0
    return X, Y_set


def save_weights(multilayer):
    """
    Save the weights of the multilayer-perceptron model to a csv file.
    """
    with open('weights.txt', 'w') as f:
        f.write(str(multilayer.get_size() - 1) + '\n')
        for layer in multilayer.get_layers()[1:]:
            f.write(str(layer.get_neurons()) + ' ' + str(layer.get_dendrites()) + '\n')
            for perceptron in layer.get_perceptrons():
                for weight in perceptron.get_weights():
                    f.write(str(weight) + ' ')
                f.write('\n')
    print("\033[1mDone! weights.txt file has been created and saved.\n\033[0m")
    return

if __name__ == '__main__':
    train, test = read_dataset()
    topology = (10, 10, 2)
    hidden_weights = get_hidden_weights(topology, input_len = train.shape[1] - 1)
    multilayer = MultilayerPerceptron(train.shape[1] - 1, hidden_weights)
    X_train, Y_train = give_train_format(train, true_val = 'M', neg_val = 'B')
    multilayer.train(X_train, Y_train, alpha = 0.01, max_iter = 70)
    save_weights(multilayer)
    sys.exit(0)
