import pandas as pd
import numpy as np
import sys
import itertools

# Tree implementation
class Node:
    def __init__(self, split_variable, split_type, split_value, is_leaf = False, pred = None):
        self.left = None
        self.right = None
        self.split_variable = split_variable 
        self.split_type = split_type 
        self.split_value = split_value
        self.is_leaf = is_leaf
        self.pred = pred

    def __eq__(self, other):
        if isinstance(other, Node):
            c1 = self.split_variable == other.split_variable
            c2 = self.split_type == other.split_type
            c3 = self.split_value == other.split_value
            c4 = self.is_leaf == other.is_leaf  
            c5 = self.pred == other.pred
            return c1 and c2 and c3 and c4 and c5       
        return NotImplemented
    

def printTree(node, level=0):
    if node != None:
        if node.is_leaf:
            print(' ' * 4 * level,  '--', node.pred)
        else:
            printTree(node.left, level + 1)
            print(' ' * 4 * level,  '--', node.split_variable, node.split_type, node.split_value)
            printTree(node.right, level + 1)


# Decision Tree script implementaion

def is_categorical(d):
    return d.dtypes.name == 'category'


def gini_impurity(y):
    P = y.value_counts() / y.shape[0]
    return 1 - np.sum(P**2)


def entropy(y):
    P = y.value_counts() / y.shape[0]
    return - np.sum(P * np.log2(P + 1e-20))


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


def leaf_output(y_cluster, is_numeric):
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
    tree = None,
    max_depth = None, 
    min_samples_split = None, 
    min_info_gain = 1e-10, 
    depth = 0, 
    max_categories = 20
):
    if depth == 0:
        check_max_category(X, max_categories)

    depth_cond = True if (max_depth == None or depth < max_depth) else False
    sample_cond = True if (min_samples_split == None or X.shape[0] > min_samples_split) else False

    if depth_cond and sample_cond:
        split_variable, split_value, info_gain, is_feature_numeric = get_best_split(X, y)
        
        if (info_gain is not None and info_gain >= min_info_gain):
            left, right = make_split(X, split_variable, split_value, is_feature_numeric)
            split_type = "<=" if is_feature_numeric else "in"
            tree = Node(split_variable, split_type, split_value)
            depth += 1
            l_tree = train_tree(left, y.loc[left.index], is_y_numeric, tree, max_depth, min_samples_split, min_info_gain, depth)
            r_tree = train_tree(right, y.loc[right.index], is_y_numeric, tree, max_depth, min_samples_split, min_info_gain, depth)
            # if l == r:
            #   prune()
            tree.left = l_tree
            tree.right = r_tree
            return tree

        else:
            pred = leaf_output(y, is_y_numeric)
            return Node(None, None, None, True, pred)

    else:
        pred = leaf_output(y, is_y_numeric)
        return Node(None, None, None, True, pred)


def predict(x_i, tree):
    if tree.is_leaf:
        return tree.pred
    elif tree.split_type == "<=": #is_numeric
        if x_i[tree.split_variable] <= tree.split_value:
            return predict(x_i, tree.left)
        else:
            return predict(x_i, tree.right)
    else: 
        if x_i[tree.split_variable] in tree.split_value:
            return predict(x_i, tree.left)
        else:
            return predict(x_i, tree.right)


def print_confusion_matrix(tp, tn, fp, fn):
    s = f'''  
Confusion Matrix:

    _____|____actual_____
         | {tp}      {fp}
    pred |
         | {fn}      {tn}
    '''
    print(s)


def eval_model(y_test, y_pred):
    str_sep = '--------\n'
    compare = list(zip(y_test, y_pred)) # Actual vs Predicted
    tp = sum(True for i in compare if i[0]==1 and i[1]==1) # true_positives (being malignant and pred malignant)
    tn = sum(True for i in compare if i[0]==0 and i[1]==0) # true_negatives (being benign and pred benign)
    fp = sum(True for i in compare if i[0]==0 and i[1]==1) # false_positives (being benign and pred malignant)
    fn = sum(True for i in compare if i[0]==1 and i[1]==0) # false_negatives (being malignant and pred benign)
    print(str_sep)
    print_confusion_matrix(tp, tn, fp, fn)
    # Accuracy
    acc = (tp+tn) / (tp+fp+tn+fn)
    print(str_sep, 'Accuracy:\t', acc)
    # Precision
    prec = tp / (tp + fp)
    print(str_sep, 'Precision:\t', prec)
    # Recall aka Sensitivity
    rec = tp / (tp + fn)
    print(str_sep, 'Recall:\t', rec)
    # Specificity
    spec = tn / (tn + fp)
    print(str_sep, 'Specifity:\t', spec)
    # F1-score
    f1 = 2 * ( prec * rec ) / (prec + rec)
    print(str_sep, 'F1- Score:\t', f1)


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
    # Hyperparameters
    max_depth = 1
    min_samples_split = None
    min_information_gain  = 1e-5
    # Train + pred + eval
    tree = train_tree(X_train, y_train, False, max_depth, min_samples_split, min_information_gain)
    if max_depth <= 2:
        printTree(tree)
    predictions = []
    for _, row in X_test.iterrows():
        predictions.append(predict(row, tree))
    eval = pd.DataFrame({'actual': y_test, 'pred': predictions})
    eval_model(eval['actual'], eval['pred'])


if __name__ == "__main__":
    main()