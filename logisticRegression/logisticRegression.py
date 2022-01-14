import numpy as np
import sys


def sigmoid(z): 
    return 1 / (1 + np.exp(-z)) - 0.00000001

def predict(X, weights):
    z = np.dot(weights, X.T)
    return sigmoid(z)

def cost(X, y, weights):
    y_pred = predict(X, weights)
    m = len(X)
    return -(1 / m) * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def update_weights(X, y, weights, learning_rate):
    m = len(X)
    y_pred = predict(X, weights)
    gradient = np.dot(X.T,  y_pred - y)
    gradient /= m
    gradient *= learning_rate
    weights -= gradient
    return weights

def train(X, y, weights, learning_rate, noEpoch):
    J = [cost(X, y, weights)] 
    for i in range(0, noEpoch):
        weights = update_weights(X, y, weights, learning_rate)
        J.append(cost(X, y, weights))
    return J, weights

def predict2(X, y, weights, learning_rate, epochs):
    J, weights = train(X, y, weights, learning_rate, epochs) 
    h = predict(X, weights)
    for i in range(len(h)):
        h[i] = 1 if h[i] >=0.5 else 0
    y = list(y)
    acc = np.sum([y[i] == h[i] for i in range(len(y))])/len(y)
    return J, acc


def main():
    if len(sys.argv) != 2:
        print('Please specify (only) one argument i.e. the csv file')
        sys.exit()


if __name__ == "__main__":
    main()