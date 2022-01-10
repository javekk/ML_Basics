import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def plot_regression_line(X_test, y_test, y_pred):
    plt.scatter(X_test, y_test, color= "g", marker= "o") #plot points
    plt.plot(X_test, y_pred, color = "b") #plot regression line
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def processData():
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
    X_train, y_train, X_test, y_test = processData()
    model = linear_model.LinearRegression() # Get Model
    model.fit(X_train, y_train) # Train
    y_pred = model.predict(X_test)
    #Print useful data
    print("Mean squarred Error: %.2f" % mean_squared_error(y_test, y_pred))
    print("Variance score: %.2f" % r2_score(y_test, y_pred))
    print("Coefficients: %.2f" % model.coef_)
    #Plot
    plot_regression_line(X_test, y_test, y_pred)


if __name__ == "__main__":
    main()