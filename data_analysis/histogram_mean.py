import matplotlib.pyplot as plt
import pandas as pd
import sys

def read_dataset():
    """
    Reads the dataset with the raw data.
    """
    if len(sys.argv) != 1:
        print("\033[1m\033[91mError. histogram_mean.py does not take any argument.\n\033[0m")
        sys.exit(1)
    try:
        dataset = pd.read_csv('../datasets/dataset_clean.csv')
    except:
        print("\033[1m\033[91mError. dataset_clean.csv can't be read.\n\033[0m")
        sys.exit(1)
    return dataset

if __name__ == '__main__':
    dataset = read_dataset()
    X = dataset.iloc[:, 1:11]
    fig, axis = plt.subplots(2, 5)
    fig.suptitle('Wisconsin Diagnostic Breast Cancer (mean hist)', fontsize = 16)
    y_pos = 0
    x_pos = 0
    for label in X:
        benign = dataset.loc[dataset['diagnosis'] == 'B'].loc[:, label]
        malign = dataset.loc[dataset['diagnosis'] == 'M'].loc[:, label]
        axis[y_pos, x_pos].hist(benign, alpha = 0.75, color = 'cornflowerblue', label = 'benign')
        axis[y_pos, x_pos].hist(malign, alpha = 0.75, color = 'firebrick', label = 'malignant')
        axis[y_pos, x_pos].set_xlabel(label)
        axis[y_pos, x_pos].grid()
        x_pos += 1
        if x_pos == 5:
            y_pos += 1
            x_pos = 0
    plt.legend(['benign', 'malignant'])
    plt.show()
    sys.exit(0)
