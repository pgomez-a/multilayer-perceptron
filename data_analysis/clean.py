import pandas as pd
import sys

def read_dataset():
    """
    Reads the dataset with the raw data.
    """
    if len(sys.argv) != 1:
        print("\033[1m\033[91mError. clean.py does not take any argument.\n\033[0m")
        sys.exit(1)
    try:
        dataset = pd.read_csv('../datasets/dataset_raw.csv', header = None)
    except:
        print("\033[1m\033[91mError. dataset_raw.csv can't be read.\n\033[0m")
        sys.exit(1)
    return dataset

def get_columns(prefix, column):
    """
    Adds the given prefix to the column list.
    """
    output = list()
    for item in column:
        output.append(prefix + ' ' + item)
    return output

if __name__ == '__main__':
    raw_dataset = read_dataset()
    dataset = pd.DataFrame(raw_dataset.iloc[:, 1:], dtype = object)
    columns = ['radius', 'texture', 'perimeter', 'area', 'smoothness']
    columns += ['compactness', 'concavity', 'concave points', 'symmetry', 'fractal dimension']
    dataset_columns = ['diagnosis']
    dataset_columns += get_columns('mean', columns)
    dataset_columns += get_columns('std', columns)
    dataset_columns += get_columns('worst', columns)
    dataset.columns = dataset_columns
    dataset.to_csv('../datasets/dataset_clean.csv', index = False)
    print("\033[1mDone! dataset_clean.csv has been created and saved.\n\033[0m")
    sys.exit(0)
