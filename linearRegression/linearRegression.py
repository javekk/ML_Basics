import sys
import numpy as np
import matplotlib.pyplot as plt
import csv


def coef_estimation(x, y):
    n = np.size(x)
    mean_x, mean_y = np.mean(x), np.mean(y)
    cross_xy = np.sum(y * x) - (n * mean_y * mean_x) # cross deviation
    cross_xx = np.sum(x * x) - (n * mean_x * mean_x) # x deviation
    b_1 = cross_xy / cross_xx # slope
    b_0 = mean_y - (b_1*mean_x) # intercept
    return (b_0, b_1)


def plot_regression_line(x,y,b):
    plt.scatter(x,y, color= "g", marker= "o", s = 30) #plot points
    y_pred = b[0] + b[1]*x #predict response vector
    plt.plot(x, y_pred, color = "b") #plot regression line
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
    b = coef_estimation(x,y)
    print("b_0 = {}\nb_1 = {}".format(b[0], b[1]))
    print("Best fitting line: Y = {:.2f} + {:.2f}X".format(b[0], b[1]))
    plot_regression_line(x,y,b)


if __name__ == "__main__":
    main()