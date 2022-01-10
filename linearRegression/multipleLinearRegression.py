import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def plot_regression_line(y_test, y_pred):
    # print the error against effective and predicted values
    plt.scatter(y_pred, y_test-y_pred,  color= "r", marker= "o") 
    plt.xlabel('pred')
    plt.ylabel('error')
    plt.show()


def processData():
    # https://github.com/scikit-learn/scikit-learn/blob/0d378913b/sklearn/datasets/_base.py#L911
    diabetes_ds = datasets.load_diabetes()
    return train_test_split(diabetes_ds.data, diabetes_ds.target, test_size=0.1, random_state=1)


def main():
    X_train, X_test, y_train, y_test = processData()
    model = linear_model.LinearRegression() # Get Model
    model.fit(X_train, y_train) # Train
    y_pred = model.predict(X_test) 

    #Print useful data
    print("Mean squarred Error: %.2f" % mean_squared_error(y_test, y_pred))
    print("Variance score: %.2f" % r2_score(y_test, y_pred))
    print("Coefficients: ", model.coef_)
    #Plot
    plot_regression_line(y_test, y_pred)


if __name__ == "__main__":
    main()