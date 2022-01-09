import sys
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def plot_regression_line(X_train, y_train, X_test, y_test, y_pred):
    plt.scatter(X_train, y_train,  color= "r", marker= "x", label="train")
    plt.scatter(X_test, y_test, color= "g", marker= "o", s = 30, label="test") 
    plt.plot(X_test, y_pred, color= "b", marker= "d", label="pred")
    plt.legend(loc="upper right")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def read_data(filePath):
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader, None)
        data = np.array(list(reader)).astype(float)
        return data


def process_data(data):
    n_split = int( len(data) * .90 )
    np.random.seed(38) 
    np.random.shuffle(data)
    X_train = data[:n_split, 0, np.newaxis]
    X_test = data[n_split:, 0, np.newaxis]
    y_train = data[:n_split, 1]
    y_test = data[n_split:, 1]
    return (X_train, y_train, X_test, y_test)


def main():
    if len(sys.argv) != 2:
        print('Please specify (only) one argument i.e. the csv file')
        sys.exit()
    data = read_data(sys.argv[1])    
    X_train, y_train, X_test, y_test = process_data(data)
    model = linear_model.LinearRegression() # Get Model
    model.fit(X_train, y_train) # Train
    y_pred = model.predict(X_test)
    #Print useful data
    print("Mean squarred Error: %.2f" % mean_squared_error(y_test, y_pred))
    print("Variance score: %.2f" % r2_score(y_test, y_pred))
    print("Coefficients: %.2f" % model.coef_)
    #Plot
    plot_regression_line(X_train, y_train, X_test, y_test, y_pred)


if __name__ == "__main__":
    main()