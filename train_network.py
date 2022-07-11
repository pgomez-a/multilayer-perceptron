from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
    np.random.seed(42)
    for pos in range(len(topology)):
        if pos == 0:
            layer = np.random.rand(topology[pos], input_len) * np.sqrt(1 / (input_len))
        else:
            layer = np.random.rand(topology[pos], topology[pos - 1] + 1) * np.sqrt(1 / (topology[pos - 1] + 1))
        weights.append(layer)
    return weights

def give_backp_format(data, opt_outputs):
    """
    Formats the values that will be used to train the multilayer.
    """
    X = np.copy(data.iloc[:,1:].to_numpy())
    Y = np.copy(data.iloc[:,0].to_numpy())
    Y_format = np.zeros((Y.shape[0], opt_outputs.size))
    for row in range(Y.shape[0]):
        index = np.where(opt_outputs == Y[row])[0]
        if index.size:
            Y_format[row][index[0]] = 1.0
    return X, Y_format

def normalize_values(X_train, X_test):
    """
    Normalizes the independend variables and returns the mean, max and min.
    """
    X = np.copy(X_train.transpose())
    mean_list = np.zeros(X.shape[0])
    max_list = np.zeros(X.shape[0])
    min_list = np.zeros(X.shape[0])
    for feature in range(X.shape[0]):
        mean_list[feature] = X[feature].mean()
        max_list[feature] = X[feature].max()
        min_list[feature] = X[feature].min()
    X_train = X_train.transpose()
    X_test = X_test.transpose()
    for feature in range(X.shape[0]):
        X_train[feature] = (X_train[feature] - mean_list[feature]) / (max_list[feature] - min_list[feature])
        X_test[feature] = (X_test[feature] - mean_list[feature]) / (max_list[feature] - min_list[feature])
    X_train = X_train.transpose()
    X_test = X_test.transpose()
    return mean_list, max_list, min_list

def predict(X, multilayer, opt_outputs):
    """
    Performs a prediction for each of the given inputs.
    """
    Y_hat_num = np.zeros((X.shape[0], opt_outputs.size), dtype = float)
    Y_hat_str = np.zeros((X.shape[0]), dtype = object)
    for pos in range(X.shape[0]):
        val = multilayer.forward_propagation(X[pos])
        Y_hat_num[pos] = val
        Y_hat_str[pos] = opt_outputs[np.where(val == max(val))[0][0]]
    return Y_hat_num, Y_hat_str

def cost(Y, Y_hat, opt_outputs):
    """
    Performs a binary cross-entropy error function to evaluate the accuracy of the model.
    """
    Y = np.copy(Y.transpose())
    Y_hat = np.copy(Y_hat.transpose())
    cost = 0.0
    for option in range(opt_outputs.size):
        cost +=  -sum(Y[option] * np.log(Y_hat[option]) + (1 - Y[option]) * np.log(1 - Y_hat[option])) / Y.shape[1]
    return cost / opt_outputs.size

def display_graph(X_axis, train_cost, train_acc, test_cost, test_acc):
    """
    Displays a graph of the cost showing the learning curve of the algorithm.
    """
    fig, axis = plt.subplots(1, 2)
    axis[0].plot(X_axis, train_cost, alpha = 0.75, linewidth = 2, color = "darkred", label = "Train Cost")
    axis[0].plot(X_axis, test_cost, alpha = 0.75, linewidth = 2, color = "salmon", label = "Test Cost")
    axis[0].set_xlabel("Epochs")
    axis[0].set_ylabel("Cost")
    axis[0].set_title("Train cost: {:.4f}          Test cost: {:.4f}".format(train_cost[-1], test_cost[-1]))
    axis[0].legend()
    axis[0].grid()
    axis[1].plot(X_axis, train_acc, alpha = 0.75, linewidth = 2, color = "darkcyan", label = "Train Accuracy")
    axis[1].plot(X_axis, test_acc, alpha = 0.75, linewidth = 2, color = "skyblue", label = "Test Accuracy")
    axis[1].set_xlabel("Epochs")
    axis[1].set_ylabel("Accuracy")
    axis[1].set_title("Train accuracy: {:.4f}%          Test accuracy: {:.4f}%".format(train_acc[-1] * 100, test_acc[-1] * 100))
    axis[1].legend()
    axis[1].grid()
    plt.show()

def train_model(opt_outputs, Y_train_str, Y_test_str, X_train, Y_train, X_test, Y_test, alpha, max_iter):
    """
    Trains the multilayer-perceptron using backpropagation and gradient descent.
    """
    index_list = list()
    train_cost_list = list()
    train_acc_list = list()
    test_cost_list = list()
    test_acc_list = list()
    with open("historic.txt", "w") as f:
        f.write("An historic of the metric obtained during training:\n")
    for i in range(max_iter):
        multilayer.back_propagation(X_train, Y_train, alpha)
        Y_hat_train_num, Y_hat_train_str = predict(X_train, multilayer, opt_outputs)
        Y_hat_test_num, Y_hat_test_str = predict(X_test, multilayer, opt_outputs)
        train_cost = cost(Y_train, Y_hat_train_num, opt_outputs)
        test_cost = cost(Y_test, Y_hat_test_num, opt_outputs)
        train_acc = accuracy_score(Y_train_str, Y_hat_train_str)
        test_acc = accuracy_score(Y_test_str, Y_hat_test_str)
        index_list.append(i)
        train_cost_list.append(train_cost)
        train_acc_list.append(train_acc)
        test_cost_list.append(test_cost)
        test_acc_list.append(test_acc)
        print("epoch {}/{} - loss: {:.4f} - val_loss: {:.4f} - acc: {:.4f}% - val_acc: {:.4f}%".format(i + 1, max_iter, train_cost, test_cost, train_acc * 100, test_acc * 100))
        with open("historic.txt", "a") as f:
            f.write("epoch {}/{} - loss: {:.4f} - val_loss: {:.4f} - acc: {:.4f}% - val_acc: {:.4f}%\n".format(i + 1, max_iter, train_cost, test_cost, train_acc * 100, test_acc * 100))
    display_graph(index_list, train_cost_list, train_acc_list, test_cost_list, test_acc_list)
    return

def save_weights(multilayer, mean_list, max_list, min_list):
    """
    Save the weights of the multilayer-perceptron model to a csv file.
    """
    with open('weights.txt', 'w') as f:
        f.write(str(multilayer.get_size()) + '\n')
        for layer in multilayer.get_layers():
            f.write(str(layer.get_neurons()) + ' ' + str(layer.get_dendrites()) + '\n')
            for perceptron in layer.get_perceptrons():
                for weight in perceptron.get_weights():
                    f.write(str(weight) + ' ')
                f.write('\n')
        f.write("Mean: ")
        for val in mean_list:
            f.write(str(val) + ' ')
        f.write("\nMax: ")
        for val in max_list:
            f.write(str(val) + ' ')
        f.write("\nMin: ")
        for val in min_list:
            f.write(str(val) + ' ')
    print("\033[1m\nDone! weights.txt file has been created and saved.\n\033[0m")
    return

if __name__ == '__main__':
    train, test = read_dataset()
    opt_outputs = train.iloc[:, 0].unique()
    topology = (10, 5, opt_outputs.size)
    hidden_weights = get_hidden_weights(topology, input_len = train.shape[1] - 1)
    multilayer = MultilayerPerceptron(train.shape[1] - 1, hidden_weights)
    X_train, Y_train = give_backp_format(train, opt_outputs)
    X_test, Y_test = give_backp_format(test, opt_outputs)
    Y_train_str = np.copy(train.iloc[:, 0].to_numpy())
    Y_test_str = np.copy(test.iloc[:, 0].to_numpy())
    mean_list, max_list, min_list = normalize_values(X_train, X_test)
    train_model(opt_outputs, Y_train_str, Y_test_str, X_train, Y_train, X_test, Y_test, alpha = 0.01, max_iter = 70)
    save_weights(multilayer, mean_list, max_list, min_list)
    sys.exit(0)
