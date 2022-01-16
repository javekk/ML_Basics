import numpy as np
import sys
import csv
import matplotlib.pyplot as plt



def sigmoid(z): 
    return 1 / (1 + np.exp(-z)) - 0.00000001


def predict(X, weights):
    z = np.dot(weights, X.T)
    return sigmoid(z)


def cost(X, y, weights):
    y_pred = predict(X, weights)
    # Cross Entropy
    m = len(X) # number of samples
    class_cost_y1 = y * np.log(y_pred) # function for y = 1 
    class_cost_y0 = (1 - y) * np.log(1 - y_pred) # function for y = 0
    return -(1 / m) * np.sum(class_cost_y1 + class_cost_y0)


def update_weights(X, y, weights, learning_rate):
    m = len(X) # number of samples
    y_pred = predict(X, weights)
    gradient = np.dot(X.T,  y_pred - y) # compute derivatives
    gradient /= m # take the mean
    gradient *= learning_rate
    weights -= gradient #update weight
    return weights


def train(X, y, weights, learning_rate, epochs):
    print("Learning rate: ", learning_rate)
    print("Number of epochs: ", epochs)
    J = [cost(X, y, weights)] 
    for i in range(0, epochs):
        weights = update_weights(X, y, weights, learning_rate)
        J.append(cost(X, y, weights))
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
    data = data[:, 1:] #remove id
    y = np.where(data[:,9] == 4 , 1, 0) # benign(2) -> 0, malignant(4) -> 1
    X = data[:, :-1] #remove y
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


def plot_cost_trend(J):
    plt.scatter(range(0, len(J)), J, color= "g", marker= "o", s = 3)
    plt.title('Cost evelution')
    plt.xlabel('No. Epochs')
    plt.ylabel('Cost')
    plt.show()


def plot_decision_boundary(y_pred_cont, y_test, bound):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    t = list(zip(y_pred_cont, y_test))
    benign = [ti[0] for ti in t if ti[1] == 0]
    malignant = [ti[0] for ti in t if ti[1] == 1]
    ax.scatter([i for i in range(len(benign))], benign, s=25, c='b', marker="o", label='benign')
    ax.scatter([i for i in range(len(malignant))], malignant, s=25, c='r', marker="s", label='malignant')
    plt.legend(loc='upper right');
    ax.set_title("Predicitions")
    ax.set_xlabel('m')
    ax.set_ylabel('Probability')
    plt.axhline(bound, color='black')
    plt.show()


def compute_accuracy(y_test, y_pred):
    acc = np.sum([y_test[i] == y_pred[i] for i in range(len(y_test))])/len(y_test)
    print('--------')
    print("Accuracy on test set: ", acc)
    print('--------')


def main():
    if len(sys.argv) != 2:
        print('Please specify (only) one argument i.e. the Breast Cancer Wisconsin csv file')
        sys.exit()
    # Data processing
    X_train, y_train, X_test, y_test = data_preprocessing(read_data(sys.argv[1]))
    # Hyperparameters
    weights = np.random.rand(len(X_train[0])) 
    learing_rate = 0.001
    epochs = 20000
    bound = 0.5
    J, weights = train(X_train, y_train, weights, learing_rate, epochs) # J = cost
    plot_cost_trend(J)
    # Make predictions
    y_pred_cont = predict(X_test, weights)
    y_pred = np.where(y_pred_cont >= bound , 1, 0) 
    #plot some cool graph
    compute_accuracy(y_test, y_pred)
    plot_decision_boundary(y_pred_cont, y_test, bound)


if __name__ == "__main__":
    main()