import pandas as pd
import numpy as np
import sys



def gini_impurity(y):
    P = y.value_counts() / y.shape[0]
    return 1 - np.sum(P**2)

def entropy(y):
    P = y.value_counts() / y.shape[0]
    return - np.sum(P * np.log2(P + 0.00000001))
    

def data_preprocessing(data, split_threshold = .9):
    data = data.loc[:, 'ClumpThickness':] #remove id
    data.sample(frac=1, random_state=42).reset_index(drop=True)
    y = pd.DataFrame(np.where(data['Class'] == 4 , 1, 0)) # benign(2) -> 0, malignant(4) -> 1
    X = data.iloc[:, :-1] #remove y
    n_split = int( data.shape[0] * split_threshold ) 
    X_train = X[:n_split]
    X_test = X[n_split:]
    y_train = y[:n_split]
    y_test = y[n_split:]
    return (X_train, X_test, y_train, y_test )


def main():
    data_path = '../data/breast-cancer-wisconsin.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    # Data processing
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test  = data_preprocessing(data)
    
    print("PANE")


if __name__ == "__main__":
    main()
