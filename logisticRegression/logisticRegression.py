import numpy as np
import sys
import csv

def sigmoid(z): 
    return 1 / (1 + np.exp(-z)) - 0.00000001

def predict(X, weights):
    z = np.dot(weights, X.T)
    return sigmoid(z)

def cost(X, y, weights):
    y_pred = predict(X, weights)
    m = len(X)
    class_cost_y1 = y * np.log(y_pred)
    class_cost_y0 = (1 - y) * np.log(1 - y_pred)
    return -(1 / m) * np.sum(class_cost_y1 + class_cost_y0)

def update_weights(X, y, weights, learning_rate):
    m = len(X)
    y_pred = predict(X, weights)
    gradient = np.dot(X.T,  y_pred - y)
    gradient /= m
    gradient *= learning_rate
    weights -= gradient
    return weights

def train(X, y, weights, learning_rate, epochs):
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
        headers = next(reader, None)
        data = np.array(list(reader)).astype(float)
        return data

def data_preprocessing(data):
    data = data[:, 1:] #remove id
    y = np.where(data[:,9] == 4 , 1, 0) # benign(2) -> 0, malignant(4) -> 1
    data = data[:, :-1] #remove y
    return data, y

def main():
    if len(sys.argv) != 2:
        print('Please specify (only) one argument i.e. the Breast Cancer Wisconsin csv file')
        sys.exit()

    X, y = data_preprocessing(read_data(sys.argv[1]))

    weights = np.random.rand(len(X[0]))
    learing_rate = 0.0001
    epochs = 100000

    J, weights = train(X, y, weights, learing_rate, epochs) 
    h = predict(X, weights)
    for i in range(len(h)):
        h[i] = 1 if h[i] >=0.5 else 0
    y = list(y)
    acc = np.sum([y[i] == h[i] for i in range(len(y))])/len(y)


if __name__ == "__main__":
    main()