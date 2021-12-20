import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def coef_estimation(x, y):
    n = np.size(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    cross_xy = np.sum(y * x) - (n * mean_y * mean_x) # cross deviation
    cross_xx = np.sum(x * x) - (n * mean_x * mean_x) # x deviation
    b_1 = cross_xy / cross_xx # slope
    b_0 = mean_y - (b_1*mean_x) # intercept
    return (b_0, b_1)


def plot_regression_line(X_test, y_test, y_pred):
    plt.scatter(X_test, y_test, color= "g", marker= "o", s = 30) #plot points
    plt.plot(X_test, y_pred, color = "b") #plot regression line
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def readData():
    # https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/datasets/_base.py#L911
    diabetes_ds = datasets.load_diabetes()
    X = diabetes_ds.data[:, 2, np.newaxis] # using one feature
    # Split data
    n_split = int( len(X) *.90 )
    X_train = X[:n_split]
    X_test = X[n_split:]
    y_train = diabetes_ds.target[:n_split]
    y_test = diabetes_ds.target[n_split:]
    return (X_train, y_train, X_test, y_test)


def main():
    X_train, y_train, X_test, y_test = readData()
    model = linear_model.LinearRegression() # Get Model
    model.fit(X_train, y_train) # Train
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean squarred Error: %.2f" % mse)
    variace = r2_score(y_test, y_pred)
    print("Variance score: %.2f" % variace)
    coefficients = model.coef_
    print("Coefficients: %.2f" % coefficients)


    plot_regression_line(X_test, y_test, y_pred)


if __name__ == "__main__":
    main()