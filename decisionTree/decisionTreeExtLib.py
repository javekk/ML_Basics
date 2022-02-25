import pandas as pd
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def eval_model(y_test, y_pred):
    str_sep = '--------\n'
    print(str_sep, "Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(str_sep, "Classification Report:\n", classification_report(y_test, y_pred))
    print(str_sep, "Accuracy:\n", accuracy_score(y_test, y_pred))


def data_preprocessing(data, split_threshold = .9):
    data = data.loc[:, 'ClumpThickness':] #remove id
    data.sample(frac=1, random_state=42).reset_index(drop=True)
    y = data['Class'].astype('category')
    y = y.cat.codes.astype('category')
    X = data.iloc[:, :-1] #remove y
    return train_test_split(X, y, test_size= 1-split_threshold)


def main():
    data_path = '../data/breast-cancer-wisconsin.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    # Data processing
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    # Hyperparameters
    max_depth = 1
    min_samples_split = 3

    # Train + pred + eval
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, )
    tree.fit(X_train , y_train)
    predictions = tree.predict(X_test)
    eval = pd.DataFrame({'actual': y_test, 'pred': predictions})
    eval_model(eval['actual'], eval['pred'])


if __name__ == "__main__":
    main()