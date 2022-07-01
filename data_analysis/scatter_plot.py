import matplotlib.pyplot as plt
import pandas as pd
import sys

def read_dataset():
    """
     Reads the dataset with the cleaned data.
    """
    try:
        dataset = pd.read_csv('../datasets/dataset_clean.csv')
    except:
        print("\033[1m\033[91mError. dataset_clean.csv can't be read.\n\033[0m")
        sys.exit(1)
    valid_columns = dataset.iloc[:, 1:].columns
    if len(sys.argv) == 3 and sys.argv[1] in valid_columns and sys.argv[2] in valid_columns:
        xlabel = sys.argv[1]
        ylabel = sys.argv[2]
    elif len(sys.argv) == 3:
        print("\033[1m\033[91mError. [{}, {}] not in:\n\033[0m".format(sys.argv[1], sys.argv[2]))
        print("\033[1m\033[91m\t{}\n\033[0m".format(valid_columns))
        sys.exit(1)
    else:
        print("\033[1m\033[91mError. scatter_plot.py can only take two arguments from:\n\033[0m")
        print("\033[1m\033[91m\t{}\n\033[0m".format(valid_columns))
        sys.exit(1)
    return dataset, xlabel, ylabel

if __name__ == '__main__':
    dataset, xlabel, ylabel = read_dataset()
    benign = dataset.loc[dataset['diagnosis'] == 'B']
    malign = dataset.loc[dataset['diagnosis'] == 'M']
    plt.scatter(benign.loc[:, xlabel], benign.loc[:, ylabel], alpha = 0.75, color = 'cornflowerblue', label = 'benign')
    plt.scatter(malign.loc[:, xlabel], malign.loc[:, ylabel], alpha = 0.75, color = 'firebrick', label = 'malignant')
    plt.title('Wisconsin Diagnostic Breast Cancer (WDBC)')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha = 0.75)
    plt.legend()
    plt.show()
