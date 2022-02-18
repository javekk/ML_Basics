import pandas as pd
import numpy as np
import sys
import itertools



def is_categorical(d):
    return d.dtypes.name == 'category'


def gini_impurity(target):
    P = target.value_counts() / target.shape[0]
    return 1 - np.sum(P**2)


def entropy(target):
    P = target.value_counts() / target.shape[0]
    return - np.sum(P * np.log2(P + 0.00000001))
    

def variance(target):
    return target.var() if (len(target) != 1) else 0


def information_gain(target, mask, func = entropy):   
    no_true = sum(mask) 
    no_false = len(mask) - no_true 
    
    perc_true = no_true / (no_true + no_false)
    perc_false = no_false / (no_true + no_false)

    if is_categorical(target):
        return func(target) - perc_true * func(target[mask]) - perc_false * func(target[-mask])
    else:
        return variance(target) - (perc_true * variance(target[mask])) - (perc_false * variance(target[-mask]))    


def categorical_options(feature):
    no_classes = feature.unique()
    split_poss = []
    for i_class in range(0, len(no_classes)+1):
        for subset in itertools.combinations(no_classes, i_class):
            split_poss.append(list(subset))
    # Remove first and last because the split will include all values
    return split_poss[1:-1]


def max_information_gain_split(feature, target, func=entropy):

    is_numerical = False if is_categorical(feature) else True
    
    if is_numerical:
        options = feature.sort_values().unique()[1:-1]
    else: 
        options = categorical_options(feature)

    if len(options) == 0:
        return(None, None, None, False)

    split_values = []
    info_gains = [] 

    best_info_gain = 0
    best_split = []

    for s in options:
        mask = feature < s if is_numerical else feature.isin(s)
        info_gain = information_gain(target, mask, func)
        info_gains.append(info_gain)
        split_values.append(s)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_split = s

    return (best_info_gain, best_split, is_numerical, True)


def data_preprocessing(data, split_threshold = .9):
    data = data.loc[:, 'ClumpThickness':] #remove id
    data.sample(frac=1, random_state=42).reset_index(drop=True)
    y = data['Class'].astype('category')
    y = y.cat.codes.astype('category')
    X = data.iloc[:, :-1] #remove y
    n_split = int( data.shape[0] * split_threshold ) 
    X_train = X.iloc[:n_split]
    X_test = X.iloc[n_split:]
    y_train = y.iloc[:n_split]
    y_test = y.iloc[n_split:]
    return ( X_train, X_test, y_train, y_test )


def main():
    data_path = '../data/breast-cancer-wisconsin.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    # Data processing
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test  = data_preprocessing(data)
    max_information_gain_split(X_test['ClumpThickness'], y_test) 
    print("PANE")


if __name__ == "__main__":
    main()