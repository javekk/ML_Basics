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
        oneHot[i, Y[i]-1] = 1
    return oneHot


def softmax(Z):
    soft = (np.exp(Z).T / np.sum(np.exp(Z),axis=1)).T 
    return soft


def predict(X, weights):
    Z = - ( X @ weights )
    P = softmax(Z)
    return np.argmax(P,axis=1)


def cost(X, Y_1h, weights):
    Z = - (X @ weights) 
    N = X.shape[0]
    t1 = np.trace(X @ weights @ Y_1h.T)
    t2 = np.sum(np.log(np.sum(np.exp(Z), axis=1)))
    loss =  t1 + t2
    loss = loss / N
    return loss


def update_weights(X, y, weights, learning_rate):
    m = len(X) # number of samples
    Z = - ( X @ weights )
    y_pred = softmax(Z)
    gradient = (X.T @ (y - y_pred)) + 2 * weights
    gradient /= m # take the mean
    gradient *= learning_rate
    weights -= gradient #update weights
    return weights


def train(X, y, weights, learning_rate, epochs):
    print("Learning rate: ", learning_rate)
    print("Number of epochs: ", epochs)
    y_1h = one_hot_encoding(y)
    J = [cost(X, y_1h, weights)] 
    for i in range(0, epochs):
        weights = update_weights(X, y_1h, weights, learning_rate)
        J.append(cost(X, y_1h, weights))
        if i % 1000 == 0:
            print("Epoch: ", i, ", Cost: ", J[i])
    return J, weights


def read_data(filePath):
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader, None) #remove headers
        data = np.array(list(reader)).astype(float)
        return data


def data_preprocessing(data):
    np.random.seed(38) 
    np.random.shuffle(data)
    y = data[:, 0].astype(int)
    X = data[:, 1:] #remove y
    # split
    n_split = int( len(data) * .90 ) 
    X_train = X[:n_split]
    X_test = X[n_split:]
    y_train = y[:n_split]
    y_test = y[n_split:]
    return (X_train, y_train, X_test, y_test)


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
    # acc = np.sum([y_test[i] == y_pred[i] for i in range(len(y_test))])/len(y_test)
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


def main():
    data_path = '../data/wine.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    # Data processing
    X_train, y_train, X_test, y_test = data_preprocessing(read_data(data_path))
    # Hyperparameters
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train =scaler.transform(X_train)

    weights = np.random.rand(X_train.shape[1], len(np.unique(y_test)))
    learing_rate = 0.001
    epochs = 20000
    threshold = 0.5
    J, weights = train(X_train, y_train, weights, learing_rate, epochs) # J = cost
    plot_cost_trend(J)

    '''
    # Make predictions
    y_pred_prob = predict(X_test, weights)
    y_pred = np.where(y_pred_prob >= threshold , 1, 0) 
    #plot some cool graph
    eval_model(y_test, y_pred)
    plot_decision_boundary(y_pred_prob, y_test, threshold)

    '''


if __name__ == "__main__":
    main()
