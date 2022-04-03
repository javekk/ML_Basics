import pandas as pd
import numpy as np
import sys

from model.RandomForest import RandomForest



def eval_model(y_test, y_pred, no_feature):
    str_sep = '------------------------------------------------------\n'
    n = len(y_test)
    mae = np.sum(np.abs(y_test - y_pred)) / n
    print(str_sep, '\tMean Absolute Error: ', mae)
    mse = np.sum((y_test - y_pred)**2) / n
    print(str_sep, '\tMean Squared Error: ', mse)
    # R-Squared R2
    y_test_mean = np.mean(y_test)
    r2num = mse
    r2den = np.sum((y_test - y_test_mean)**2) / n
    R2 = 1 - (r2num / r2den)
    print(str_sep, '\tR-Squared (R2): ', R2)
    # Adjusted R-Squared R2
    R2adj = 1 - ( ((1-R2)*(n-1)) / (n-no_feature-1) )
    print(str_sep, '\tAdjusted R-Squared (R2): ', R2adj)
    print(str_sep)


def data_preprocessing(data, split_threshold = .9):
    data = data.loc[:, :'origin'] #remove carname
    data.sample(frac=1, random_state=42).reset_index(drop=True)
    y = data.mpg
    X = data.iloc[:, 1:] #remove y
    # split
    n_split = int( data.shape[0] * split_threshold ) 
    X_train = X[:n_split]
    X_test = X[n_split:]
    y_train = y[:n_split]
    y_test = y[n_split:]
    return (X_train, X_test, y_train, y_test )


def main():
    data_path = '../data/auto.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    # Data processing
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test  = data_preprocessing(data)
    # Hyperparameters
    max_depth = 10
    number_of_trees = 3
    min_samples_split = None
    min_information_gain  = 1e-5
    # Train + pred + eval
    reg = RandomForest(True)
    reg.fit(X_train, y_train, number_of_trees, max_depth, min_samples_split, min_information_gain)
    predictions = []
    for _, row in X_test.iterrows():
        predictions.append(reg.predict(row))
    eval_model(y_test, predictions, X_test.shape[1])


if __name__ == "__main__":
    main()