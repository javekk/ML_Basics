import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing


def one_hot_encoding(Y):
    m = len(Y)
    no_class = len(np.unique(Y))
    oneHot = np.zeros(shape=(m,no_class))
    for i in range(0, m):
        oneHot[i, Y[i]-3] = 1 # classes start from 3 (3, 4, 5, 6, 7, 8, 9)
        #oneHot[i, Y[i]] = 1
    return oneHot


def softmax(Z):
    return (np.exp(Z).T / np.sum(np.exp(Z),axis=1)).T 


def predict_(X, weights):
    Z = X @ weights
    P = softmax(Z)
    return P


def predict(X, weights):
    P = predict_(X, weights)
    return np.argmax(P,axis=1) 


def cost(X, Y_1h, weights):
    P = predict_(X, weights)
    m = X.shape[0]
    return - (np.sum( Y_1h * np.log(P))) / m 
    

def update_weights(X, y_1h, weights, learning_rate):
    m = len(X) 
    P = predict_(X, weights)
    gradient = - (np.dot( X.T, ( y_1h - P ))) / m   
    weights -= gradient * learning_rate
    return weights


def train(X, y, weights, learning_rate, epochs):
    print("Learning rate: ", learning_rate)
    print("Number of epochs: ", epochs)
    y_1h = one_hot_encoding(y)
    J = [cost(X, y_1h, weights)] 
    for i in range(0, epochs):
        weights = update_weights(X, y_1h, weights, learning_rate)
        J.append(cost(X, y_1h, weights))
        if i % 100 == 0:
            print("Epoch: ", i, ", Cost: ", J[i])
    return J, weights


def read_data(filePath):
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader, None) #remove headers
        data = np.array(list(reader)).astype(float)
        return data


def standardization(X):
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / std


def normalized(X):
    min = np.min(X)
    max = np.max(X)
    return (X - min) / (max - min)


def data_preprocessing(data, rescaleType='#'):
    np.random.seed(38) 
    np.random.shuffle(data)
    y = data[:, 11].astype(int)
    X = data[:, :11] #remove y
    if rescaleType == 'N':
        X = normalized(X)
    elif rescaleType == 'S':
        X = standardization(X)
    # split
    n_split = int( len(data) * .85 ) 
    X_train = X[:n_split]
    X_test = X[n_split:]
    y_train = y[:n_split]
    y_test = y[n_split:]
    return (X_train, y_train, X_test, y_test)

'''
def data_preprocessing(data, rescaleType='#'):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris_data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.33, random_state=42)
    return (X_train, y_train, X_test, y_test)
'''

def plot_cost_trend(J):
    plt.scatter(range(0, len(J)), J, color= "g", marker= "o", s = 3)
    plt.title('Cost evelution')
    plt.xlabel('No. Epochs')
    plt.ylabel('Cost')
    plt.show()


def plot_decision_boundary(y_pred_prob, y_test, threshold):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = list(zip(y_pred_prob, y_test))
    benign = [ti[0] for ti in t if ti[1] == 0]
    malignant = [ti[0] for ti in t if ti[1] == 1]
    ax.scatter([i for i in range(len(benign))], benign, s=25, c='b', marker="o", label='benign')
    ax.scatter([i for i in range(len(malignant))], malignant, s=25, c='r', marker="s", label='malignant')
    plt.legend(loc='upper right');
    ax.set_title("Predicitions")
    ax.set_xlabel('m')
    ax.set_ylabel('Probability')
    plt.axhline(threshold, color='black')
    plt.show()


def print_confusion_matrix(tp, tn, fp, fn):
    s = f'''  
Confusion Matrix: TBD
    '''
    print(s)


def eval_model(y_test, y_pred):
    str_sep = '--------\n'
    print(str_sep)
    print(y_pred[0:30] + 4)
    print(y_test[0:30])
    '''
    # Accuracy
    print(str_sep, 'Accuracy:\t', "TBD")
    # Precision
    print(str_sep, 'Precision:\t', "TBD")
    # Recall aka Sensitivity
    print(str_sep, 'Recall:\t', "TBD")
    # Specificity
    print(str_sep, 'Specifity:\t', "TBD")
    # F1-score
    print(str_sep, 'F1- Score:\t', "TBD")
    '''


def main():
    data_path = '../data/winequality-red.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    # Data processing
    X_train, y_train, X_test, y_test = data_preprocessing(read_data(data_path), rescaleType='S')
    # Hyperparameters
    weights = np.random.rand(X_train.shape[1], len(np.unique(y_train)))
    learing_rate = 0.01
    epochs = 10000
    J, weights = train(X_train, y_train, weights, learing_rate, epochs) # J = cost
    plot_cost_trend(J)
    # Make predictions
    y_pred = predict(X_test, weights)
    #plot some cool graph
    eval_model(y_test, y_pred)
    #plot_decision_boundary(y_pred_prob, y_test, threshold)



if __name__ == "__main__":
    main()
