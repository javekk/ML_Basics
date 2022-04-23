import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def data_preprocessing(data, split_threshold = .9):
    data.drop(['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total'], axis=1, inplace=True)
    data.sample(frac=1, random_state=42).reset_index(drop=True)
    y = data['is_spam'].astype('category') # 1 spam, 0 normal
    X = data.iloc[:, :-1] #remove y
    n_split = int( data.shape[0] * split_threshold ) 
    X_train = X.iloc[:n_split]
    X_test = X.iloc[n_split:]
    y_train = y.iloc[:n_split]
    y_test = y.iloc[n_split:]
    return ( X_train, X_test, y_train, y_test )


def main():
    data_path = '../data/spambase.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    data = pd.read_csv(data_path)


if __name__ == "__main__":
    main()