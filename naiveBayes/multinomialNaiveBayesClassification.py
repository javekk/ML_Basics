import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def data_preprocessing(data, split_threshold=.9):
    data.sample(frac=1, random_state=42).reset_index(drop=True)
    y = data['is_spam'].map({'spam': 1, 'ham': 0}).astype('category')  # 1 spam, 0 normal
    X = data['sms']
    n_split = int(data.shape[0] * split_threshold)
    X_train = X.iloc[:n_split]
    X_test = X.iloc[n_split:].reset_index(drop=True)
    y_train = y.iloc[:n_split]
    y_test = y.iloc[n_split:].reset_index(drop=True)
    return (X_train, X_test, y_train, y_test)


def tokenize(s):
    return s.lower().split()


def word_count(X, y):
    counts = {}
    for s, clazz in zip(X,y):
        for token in tokenize(s):
            if token not in counts:
                counts[token] = [0, 0]
            counts[token][clazz] += 1
    return counts


def count_by_class(y):
    counts = y.value_counts()
    return (counts[0], counts[1])


def prior_prob(X, y):
    n = len(X)
    n_not_spam, n_spam = count_by_class(y)
    not_spam_prior = n_not_spam / n
    is_spam_prior = n_spam / n
    return not_spam_prior, is_spam_prior


def word_probs(counts, n_not_spam, n_is_spam):
    sm = 0.5  # smoothing term
    probs = {}
    for token, (c0, c1) in counts.items():
        p0 = (c0 + sm) / (n_not_spam + 2 * sm)
        p1 = (c1 + sm) / (n_is_spam + 2 * sm)
        probs[token] = [p0, p1]
    return probs


def predict(X, probs, not_spam_prior, is_spam_prior):
    y_pred = []
    for s in X:
        log_p0 = log_p1 = 0.0
        tokens = tokenize(s)
        for word, (p0, p1) in probs.items():
            if word in tokens:
                log_p0 += np.log(p0)
                log_p1 += np.log(p1)
        pred_not_spam = not_spam_prior * np.exp(log_p0)
        pred_is_spam = is_spam_prior * np.exp(log_p1)
        # Append also probability 
        if pred_not_spam >= pred_is_spam:
            y_pred.append([0, pred_not_spam])
        else:
            y_pred.append([1, pred_is_spam])
    return pd.DataFrame(y_pred)


def plot_decision_boundary(y_pred_prob, y_test):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    is_not_spam_list = (y_test == 0) # array with True or False, True = is Not spam
    is_spam_list = (y_test == 1) # array with True or False, True = is spam
    not_spam_probs = y_pred_prob[is_not_spam_list] 
    one_less_is_spam_prob = 1 - y_pred_prob[is_spam_list]
    is_spam_idx = [i for i in range(len(is_spam_list)) if is_spam_list[i]] # array with only indeces for not spam 
    not_spam_idx = [i for i in range(len(is_not_spam_list)) if is_not_spam_list[i]] # array with only indeces for spam 
    ax.scatter(is_spam_idx, one_less_is_spam_prob, s=5, c='r', marker="x", label='spam')
    ax.scatter(not_spam_idx, not_spam_probs, s=5, c='b', marker="o", label='normal')
    plt.legend(loc='center right');
    ax.set_title("Predicitions")
    ax.set_xlabel('m')
    ax.set_ylabel('Probability')
    plt.axhline(.5, color='black')
    plt.show()


def print_confusion_matrix(tp, tn, fp, fn):
    s = f'''  
Confusion Matrix:

    _____|____actual_____
         | {tp}      {fp}
    pred |
         | {fn}      {tn}
    '''
    print(s)


def eval_model(y_test, y_pred):
    str_sep = '--------\n'
    compare = list(zip(y_test, y_pred)) 
    tp = sum(True for i in compare if i[0]==1 and i[1]==1) 
    tn = sum(True for i in compare if i[0]==0 and i[1]==0) 
    fp = sum(True for i in compare if i[0]==0 and i[1]==1) 
    fn = sum(True for i in compare if i[0]==1 and i[1]==0) 
    print(str_sep)
    print_confusion_matrix(tp, tn, fp, fn)
    # Accuracy
    acc = (tp+tn) / (tp+fp+tn+fn)
    print(str_sep, 'Accuracy:\t', acc)
    # Precision
    prec = tp / (tp + fp)
    print(str_sep, 'Precision:\t', prec)
    # Recall aka Sensitivity
    rec = tp / (tp + fn)
    print(str_sep, 'Recall:\t', rec)
    # Specificity
    spec = tn / (tn + fp)
    print(str_sep, 'Specifity:\t', spec)
    # F1-score
    f1 = 2 * ( prec * rec ) / (prec + rec)
    print(str_sep, 'F1- Score:\t', f1)


def main():
    data_path = '../data/SMSSpamCollection.csv'
    if len(sys.argv) == 2:
        data_path = sys.argv[1]
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    # Train
    n_not_spam_, n_is_spam = count_by_class(y_train)
    not_spam_prior, is_spam_prior = prior_prob(X_train, y_train)
    counts = word_count(X_train, y_train)
    probs = word_probs(counts, n_not_spam_, n_is_spam)
    # Predict
    y_pred_and_probs = predict(X_test, probs, not_spam_prior, is_spam_prior)
    y_pred = y_pred_and_probs.iloc[:, 0]
    y_pred_prob = y_pred_and_probs.iloc[:, 1]
    #Eval
    print(' '.join([str(n) for n in y_test[0:20]]))
    print(' '.join([str(n) for n in y_pred[0:20]]))
    eval_model(y_test, y_pred)
    plot_decision_boundary(y_pred_prob, y_test)


if __name__ == "__main__":
    main()
