from sklearn.metrics import accuracy_score
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
            mean_list = np.array(f.readline().split()[1:], dtype = float)
            max_list = np.array(f.readline().split()[1:], dtype = float)
            min_list = np.array(f.readline().split()[1:], dtype = float)
    except:
        print("\033[1m\033[91mError: read_dataset. {} can't be read.\n\033[0m".format(sys.argv[2]))
        sys.exit(1)
    return weights, mean_list, max_list, min_list

def give_backp_format(data, true_val, neg_val):
    """
    Formats the values that will be used to train the multilayer.
    """
    X = np.copy(data.iloc[:,1:].to_numpy())
    Y = np.copy(data.iloc[:,0].to_numpy())
    Y[Y == true_val] = 1.0
    Y[Y == neg_val] = 0.0
    Y_format = np.zeros((Y.shape[0], 2))
    for row in range(Y.shape[0]):
        if Y[row] == 1.0:
            Y_format[row][0] = 1.0
        else:
            Y_format[row][1] = 1.0
    return X, Y_format

def normalize_values(X, mean_list, max_list, min_list):
    """
    Normalizes the independend variables.
    """
    X = X.transpose()
    for feature in range(X.shape[0]):
        X[feature] = (X[feature] - mean_list[feature]) / (max_list[feature] - min_list[feature])
    X = X.transpose()
    return

def predict(X, multilayer, true_val, neg_val):
    """
    Performs a prediction for each of the given inputs.
    """
    print("\033[1mComputing predictions...\033[0m")
    Y_hat_num = np.zeros((X.shape[0], 2), dtype = float)
    Y_hat_str = np.zeros((X.shape[0]), dtype = object)
    for pos in range(X.shape[0]):
        val = multilayer.forward_propagation(X[pos])
        Y_hat_num[pos] = val
        Y_hat_str[pos] = true_val if val[0] > val[1] else neg_val
    return Y_hat_num, Y_hat_str

def cost(Y, Y_hat):
    """
    Performs a binary cross-entropy error function to evaluate the accuracy of the model.
    """
    Y = np.copy(Y.transpose())
    Y_hat = np.copy(Y_hat.transpose())
    label_one_cost =  -sum(Y[0] * np.log(Y_hat[0]) + (1 - Y[0]) * np.log(1 - Y_hat[0])) / Y.shape[1]
    label_two_cost =  -sum(Y[1] * np.log(Y_hat[1]) + (1 - Y[1]) * np.log(1 - Y_hat[1])) / Y.shape[1]
    error = (label_one_cost + label_two_cost) / 2
    print("\033[1m\nBinary cross-entropy error value is: {:.4f}.\033[0m".format(error))
    return

if __name__ == '__main__':
    dataset = read_dataset()
    weights, mean_list, max_list, min_list = get_weights()
    multilayer = MultilayerPerceptron(dataset.shape[1] - 1, weights)
    Y_str = dataset.iloc[:, 0].to_numpy()
    X, Y_num = give_backp_format(dataset, true_val = 'M', neg_val = 'B')
    normalize_values(X, mean_list, max_list, min_list)
    Y_hat_num, Y_hat_str = predict(X, multilayer, true_val = 'M', neg_val = 'B')
    cost(Y_num, Y_hat_num)
    print("\033[1m\nThe accuracy is: {:.4f}%\n\033[0m".format(accuracy_score(Y_str, Y_hat_str) * 100))
    sys.exit(0)
