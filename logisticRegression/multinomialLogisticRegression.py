import numpy as np
import sys
import csv
import matplotlib.pyplot as plt
import seaborn as sns


def one_hot_encoding(Y):
    m = len(Y)
    no_class = len(np.unique(Y))
    oneHot = np.zeros(shape=(m,no_class))
    for i in range(0, m):
        oneHot[i, Y[i]] = 1 
    return oneHot


def softmax(Z):
    return (np.exp(Z).T / np.sum(np.exp(Z),axis=1)).T 


def predict_(X, weights):
    Z = X @ weights
    P = softmax(Z)
    return P


def predict(X, weights):
    P = predict_(X, weights)
    return np.argmax(P,axis=1) 


def cost(X, Y_1h, weights):
    P = predict_(X, weights)
    m = X.shape[0]
    return - (np.sum( Y_1h * np.log(P))) / m 
    

def update_weights(X, y_1h, weights, learning_rate):
    m = len(X) 
    P = predict_(X, weights)
    gradient = - (np.dot( X.T, ( y_1h - P ))) / m   
    weights -= gradient * learning_rate
    return weights


def train(X, y, weights, learning_rate, epochs):
    print("Learning rate: ", learning_rate)
    print("Number of epochs: ", epochs)
    y_1h = one_hot_encoding(y)
    J = [cost(X, y_1h, weights)] 
    for i in range(0, epochs):
        weights = update_weights(X, y_1h, weights, learning_rate)
        J.append(cost(X, y_1h, weights))
        if i % 100 == 0:
            print("Epoch: ", i, ", Cost: ", J[i])
    return J, weights


def read_data(filePath):
    with open(filePath, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader, None)
        data = np.array(list(reader)).astype(float)
        plotDataDistribution(data, headers)
        return data


def standardization(X):
    # rescales data to have a mean of 0 and standard deviation of 1
    mean = np.mean(X)
    std = np.std(X)
    return (X - mean) / std


def normalized(X):
    # Rescales the values into a range of [0,1] 
    min = np.min(X)
    max = np.max(X)
    return (X - min) / (max - min)


def data_preprocessing(data, rescaleType='#'):
    np.random.seed(38) 
    np.random.shuffle(data)
    y = data[:, 0].astype(int)
    y -= 1 
    X = data[:, 0:] #remove y
    if rescaleType == 'N':
        X = normalized(X)
    elif rescaleType == 'S':
        X = standardization(X)
    # split
    n_split = int( len(data) * .85 ) 
    X_train = X[:n_split]
    X_test = X[n_split:]
    y_train = y[:n_split]
    y_test = y[n_split:]
    return (X_train, y_train, X_test, y_test)


def plotDataDistribution(data, headers):
    m = np.sqrt(data.shape[1]).astype(int) + 1
    fig = plt.figure(figsize=[10.4, 8.8])
    for i, col_name in enumerate(headers):
        fig.subplots_adjust(hspace=.4, wspace=.5)
        ax = fig.add_subplot( m, m, i+1)
        p = sns.histplot(data=data, x=data[:,i], ax=ax)
        p.set_xlabel(col_name, fontsize = 10)
        p.set_ylabel("", fontsize = 10)
    plt.show()


def plot_cost_trend(J):
    plt.scatter(range(0, len(J)), J, color= "g", marker= "o", s = 3)
    plt.title('Cost evelution')
    plt.xlabel('No. Epochs')
    plt.ylabel('Cost')
    plt.show()


def print_confusion_matrix(y_test, y_pred):
    no_class = len(np.unique(y_test))
    M = np.zeros((no_class,no_class))
    for i,y in enumerate(y_test):
        M[y, y_pred[i]] += 1
    print(M)


def plot_accuracy_graph(y_pred_prob, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y_pred_prob_vec = [y_pred_prob[i,yi] for i,yi in enumerate(y_test)]  
    markers_sample = [".", "v", "1"]
    colors_sample =  ["b", "g", "r"]
    ax_colors = [colors_sample[i] for i in y_test]
    ax_y = [i for i in range(len(y_test))]
    ax.scatter(ax_y, y_pred_prob_vec, s=25, c=ax_colors, marker="o")
    plt.legend(loc='upper right');
    ax.set_title("Predicitions")
    ax.set_xlabel('m')
    ax.set_ylabel('Probability')
    plt.show()


def eval_model(y_test, y_pred):
    str_sep = '--------\n'
    print(str_sep, 'Confusion Matrix:\t')
    # Correct guess percetage
    m = y_test.shape[0]
    corrects = sum(y_test == y_pred) / m
    corr_perc = corrects * 100
    print(str_sep, 'Perc. of correct guess:\t', corr_perc)
    # Confusiong Marrix
    print_confusion_matrix(y_test, y_pred)


def main():
    data_path = '../data/wine.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    # Data processing
    X_train, y_train, X_test, y_test = data_preprocessing(read_data(data_path), rescaleType='S')
    # Hyperparameters
    weights = np.random.rand(X_train.shape[1], len(np.unique(y_train)))
    learing_rate = 0.01
    epochs = 10000
    J, weights = train(X_train, y_train, weights, learing_rate, epochs) # J = cost
    plot_cost_trend(J)
    # Make predictions
    y_pred_prod = predict_(X_test, weights)
    y_pred = np.argmax(y_pred_prod, axis=1) 
    #Eval and plot some cool graph
    eval_model(y_test, y_pred)
    plot_accuracy_graph(y_pred_prod, y_test)



if __name__ == "__main__":
    main()
