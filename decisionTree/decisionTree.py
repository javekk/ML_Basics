import pandas as pd
import numpy as np
import sys
import itertools



def is_categorical(d):
    return d.dtypes.name == 'category'


def gini_impurity(y):
    P = y.value_counts() / y.shape[0]
    return 1 - np.sum(P**2)


def entropy(y):
    P = y.value_counts() / y.shape[0]
    return - np.sum(P * np.log2(P + 0.00000001))


def variance(y):
    return y.var() if (len(y) != 1) else 0


def information_gain(y, mask, func = entropy):   
    no_true = sum(mask) 
    no_false = len(mask) - no_true 
    perc_true = no_true / (no_true + no_false)
    perc_false = no_false / (no_true + no_false)

    if is_categorical(y):
        return func(y) - perc_true * func(y[mask]) - perc_false * func(y[-mask])
    else:
        return variance(y) - (perc_true * variance(y[mask])) - (perc_false * variance(y[-mask]))    


def categorical_options(y):
    no_classes = y.unique()
    split_poss = []
    for i_class in range(0, len(no_classes)+1):
        for subset in itertools.combinations(no_classes, i_class):
            split_poss.append(list(subset))
    # Remove first and last because the split will include all values
    return split_poss[1:-1]


def max_information_gain_split(feature, y, func=entropy):
    is_numerical = False if is_categorical(feature) else True
    
    if is_numerical:
        options = feature.sort_values().unique()[1:-1]
    else: 
        options = categorical_options(feature)

    if len(options) == 0:
        return (None, None, None, False)

    split_values = []
    info_gains = [] 
    best_info_gain = 0
    best_split = []

    for s in options:
        mask = feature < s if is_numerical else feature.isin(s)
        info_gain = information_gain(y, mask, func)
        info_gains.append(info_gain)
        split_values.append(s)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_split = s

    return (best_info_gain, best_split, is_numerical, True)


def get_best_split(X, y):
    masks = X.apply(max_information_gain_split, y = y).T
    masks.columns = ['info_gain', 'split', 'is_numerical', 'is_usable_as_split' ]

    if sum(masks.is_usable_as_split) == 0: # Check if we have no splits left
        return(None, None, None, None)
    else:
        masks = masks.loc[masks.is_usable_as_split]
        split_index = masks.info_gain.astype('float').argmax()
        split_value = masks.iloc[split_index, 1] 
        info_gain = masks.iloc[split_index, 0]
        is_numeric = masks.iloc[split_index, 2]
        split_variable = masks.index[split_index]
        return(split_variable, split_value, info_gain, is_numeric)


def make_split(data, split_variable, split_value, is_numeric):
    if is_numeric:
        mask = data[split_variable] < split_value
        left = data[mask]
        right = data[-mask]
    else:
        mask = data[split_variable].isin(split_value)
        left = data[mask]
        right = data[-mask]
    return(left,right)


def prediction(y_cluster, is_numeric):
  return y_cluster.mean() if is_numeric else y_cluster.value_counts().idxmax()


def check_max_category(data, max_categories):
    check_columns = data.dtypes[data.dtypes == "category"]
    for column in check_columns:
        var_length = len(data.loc[column].unique()) 
        if var_length > max_categories:
            raise ValueError('Column ' + column + ' has '+ str(var_length) + ' unique values, max = ' +  str(max_categories))


def train_tree(
    X, 
    y, 
    is_y_numeric, 
    max_depth = None, 
    min_samples_split = None, 
    min_info_gain = 1e-10, 
    iteration = 0, 
    max_categories = 20
):
    if iteration == 0:
        check_max_category(X, max_categories)

    depth_cond = True if (max_depth == None or iteration < max_depth) else False
    sample_cond = True if (min_samples_split == None or X.shape[0] > min_samples_split) else False

    if depth_cond and sample_cond:
        split_variable, split_value, info_gain, is_feature_numeric = get_best_split(X, y)

        if (info_gain is not None and info_gain >= min_info_gain):

            iteration += 1
            left, right = make_split(X, split_variable, split_value, is_feature_numeric)
            
            # Instantiate sub-tree
            split_type = "<=" if is_feature_numeric else "in"
            question =  "{} {}  {}".format(split_variable, split_type, split_value)
            # question = "\n" + counter*" " + "|->" + var + " " + split_type + " " + str(val) 
            subtree = {question: []}

            # Find answers (recursion)
            yes_answer = train_tree(left, y.loc[left.index] , is_y_numeric, max_depth,min_samples_split, min_info_gain, iteration)
            no_answer = train_tree(right, y.loc[right.index] , is_y_numeric, max_depth,min_samples_split, min_info_gain, iteration)

            if yes_answer == no_answer:
                subtree = yes_answer
            else:
                subtree[question].append(yes_answer)
                subtree[question].append(no_answer)

        # If it doesn't match IG condition, make prediction
        else:
            pred = prediction(y, is_y_numeric)
            return pred

    # Drop dataset if doesn't match depth or sample conditions
    else:
        pred = prediction(y, is_y_numeric)
        return pred

    return subtree



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
    get_best_split(X_test, y_test) 
    max_depth = None
    min_samples_split = None
    min_information_gain  = 1e-5
    decisiones = train_tree(X_train, y_train, False, max_depth, min_samples_split, min_information_gain)
    decisiones
    print("PANE")


if __name__ == "__main__":
    main()