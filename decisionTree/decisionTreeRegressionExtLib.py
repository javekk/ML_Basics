import pandas as pd
import numpy as np
import sys
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def eval_model(y_test, y_pred, no_feature):
    str_sep = '------------------------------------------------------\n'
    print(str_sep, '\tMean Absolute Error: ', mean_absolute_error(y_test, y_pred))
    print(str_sep, '\tMean Squared Error: ', mean_squared_error(y_test, y_pred))
    print(str_sep, '\tR-Squared (R2): ', r2_score(y_test, y_pred))
    n = len(y_test)
    R2adj = 1 - ( ((1-r2_score(y_test, y_pred))*(n-1)) / (n-no_feature-1) )
    print(str_sep, '\tAdjusted R-Squared (R2): ', R2adj)
    print(str_sep)


def data_preprocessing(data, split_threshold = .9):
    data = data.loc[:, :'origin'] #remove carname
    data.sample(frac=1, random_state=42).reset_index(drop=True)
    y = data.mpg
    X = data.iloc[:, 1:] #remove y
    return train_test_split(X, y, test_size= 1-split_threshold)


def main():
    data_path = '../data/auto.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    # Data processing
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    # Hyperparameters
    max_depth = 1
    min_samples_split = 3
    # Train + pred + eval
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, )
    tree.fit(X_train , y_train)
    predictions = tree.predict(X_test)
    eval_model(y_test, predictions,  X_test.shape[1])


if __name__ == "__main__":
    main()