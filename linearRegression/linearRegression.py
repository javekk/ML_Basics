import sys
import numpy as np
import matplotlib.pyplot as plt
import csv


def coefs_estimation(x, y):
    n = np.size(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    cross_xy = np.sum(y * x) - (n * mean_y * mean_x) # cross deviation
    cross_xx = np.sum(x * x) - (n * mean_x * mean_x) # x squared deviations
    b = cross_xy / cross_xx # slope
    a = mean_y - (b*mean_x) # intercept
    return (a, b)


def plot_regression_line(x, y, a, b):
    plt.scatter(x,y, color= "g", marker= "o", s = 30) #plot points
    y_pred = a + b*x #predict response vector
    plt.plot(x, y_pred, color = "b") # plot regression line
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def read_data(filePath):
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader, None)
        data = np.array(list(reader)).astype(float).transpose()
        return (data[0], data[1])


def main():
    if len(sys.argv) != 2:
        print('Please specify (only) one argument i.e. the csv file')
        sys.exit()
    x, y = read_data(sys.argv[1])
    a, b = coefs_estimation(x,y)
    print("a = {}\nb = {}".format(a, b))
    print("Best fitting line: Y = {:.2f} + {:.2f}X".format(a, b))
    plot_regression_line(x,y,a,b)


if __name__ == "__main__":
    main()