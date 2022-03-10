import pandas as pd
import numpy as np
import sys

from model.DecisionTree import DecisionTree


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
    max_depth = 25
    min_samples_split = None
    min_information_gain  = 1e-5
    # Train + pred + eval
    tree = DecisionTree()
    tree.fit(X_train, y_train, False, max_depth, min_samples_split, min_information_gain)
    if max_depth <= 3:
        tree.model.printTree()
    predictions = []
    for _, row in X_test.iterrows():
        predictions.append(tree.predict(row))
    eval = pd.DataFrame({'actual': y_test, 'pred': predictions})
    eval_model(eval['actual'], eval['pred'])


if __name__ == "__main__":
    main()