import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def separate_by_classes(X, y):
    classes = np.unique(y)
    class_dfs = {}
    for clazz in classes:
        mask = (y == clazz)
        class_dfs[clazz] = X[mask]
    return class_dfs


def get_class_frequences(class_dfs, n):
    class_frequency = {}
    for clazz, data in class_dfs.items():
        class_frequency[clazz] = len(data) / n
    return class_frequency


def get_class_mean(class_dfs):
    class_means = {}
    for clazz, _ in class_dfs.items():
        class_means[clazz] = class_dfs[clazz].mean()
    return class_means


def get_class_std(class_dfs):
    class_stds = {}
    for clazz, _ in class_dfs.items():
        class_stds[clazz] = class_dfs[clazz].std()
    return class_stds


def data_preprocessing(data, split_threshold = .9):
    data = data.loc[:, 'ClumpThickness':] #remove id
    data.sample(frac=1, random_state=42).reset_index(drop=True)
    y = data['Class'].map({4: 1, 2: 0}).astype('category') # 4 being malignant, 2 being benign
    y = y.cat.codes.astype('category')
    X = data.iloc[:, :-1] #remove y
    n_split = int(data.shape[0] * split_threshold) 
    X_train = X.iloc[:n_split]
    X_test = X.iloc[n_split:]
    y_train = y.iloc[:n_split]
    y_test = y.iloc[n_split:]
    return (X_train, X_test, y_train, y_test)


def main():
    data_path = '../data/breast-cancer-wisconsin.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    # Data processing
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    # Fit
    class_dfs = separate_by_classes(X_train, y_train)
    class_freq = get_class_frequences(class_dfs, len(X_train))
    means = get_class_mean(class_dfs)
    std = get_class_std(class_dfs)
    print("cane")



if __name__ == "__main__":
    main()
