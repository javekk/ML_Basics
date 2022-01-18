import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



def train(X, y):
    print("Training...")
    clf = LogisticRegression(random_state=0, verbose=1).fit(X, y)
    return clf


def read_data(filePath):
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader, None) #remove headers
        data = np.array(list(reader)).astype(float)
        return data


def data_preprocessing(data):
    data = data[:, 1:] #remove id
    y = np.where(data[:,9] == 4 , 1, 0) # benign(2) -> 0, malignant(4) -> 1
    X = data[:, :-1] #remove y
    # split
    p_split = .10
    # X_train, X_test, y_train, y_test
    return train_test_split(X, y, random_state=0, test_size=p_split)


def plot_decision_boundary(y_pred_cont, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = list(zip(y_pred_cont, y_test))
    benign = [ti[0][0] for ti in t if ti[1] == 0]
    malignant = [ti[0][1] for ti in t if ti[1] == 1]
    ax.scatter([i for i in range(len(benign))], benign, s=25, c='b', marker="o", label='benign')
    ax.scatter([i for i in range(len(benign),len(benign)+len(malignant))], malignant, s=25, c='r', marker="s", label='malignant')
    plt.legend(loc='lower right');
    ax.set_title("Probability of a prediction")
    ax.set_xlabel('m')
    ax.set_ylabel('Probability')
    plt.show()



def eval_model(y_test, y_pred):
    str_sep = '--------\n'
    print(str_sep, 'Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
    # Classification Report
    print(str_sep, 'Classification Report:\n', classification_report(y_test, y_pred))
    # Accuracy
    print(str_sep, 'ACCURACY:\n', accuracy_score(y_test, y_pred))


def main():
    if len(sys.argv) != 2:
        print('Please specify (only) one argument i.e. the Breast Cancer Wisconsin csv file')
        sys.exit()
    # Data processing
    X_train, X_test, y_train, y_test = data_preprocessing(read_data(sys.argv[1]))
    # Train
    clf = train(X_train, y_train)
    # Make predictions
    y_pred_prob = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test) 
    #plot some cool graph
    eval_model(y_test, y_pred)
    plot_decision_boundary(y_pred_prob, y_test)


if __name__ == "__main__":
    main()
