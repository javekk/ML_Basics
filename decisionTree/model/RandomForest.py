import numpy as np
from collections import Counter

from model.Node import Node
from model.DecisionTree import DecisionTree


class RandomForest:
    def __init__(self, is_y_numeric = False):
        self.decision_trees = []
        self.is_y_numeric = is_y_numeric
        
  
    def _sample(self, X, y):
        m = X.shape[0]
        samples = np.random.choice(a=m, size=m, replace=False)
        return X.iloc[samples], y.iloc[samples]

        
    def fit(
        self, 
        X, 
        y, 
        num_trees = 15,
        max_depth = None, 
        min_samples_split = None, 
        min_info_gain = 1e-10, 
        max_categories = 20
    ):
        if len(self.decision_trees) > 0:
            self.decision_trees = []        
        for _ in range(num_trees):
            clf = DecisionTree(self.is_y_numeric)
            _X, _y = self._sample(X, y) # Get data sample
            clf.fit(_X, _y, max_depth, min_samples_split, min_info_gain, max_categories)
            self.decision_trees.append(clf)
    

    def predict(self, X):
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
        if(self.is_y_numeric):
            counter = Counter(y)
            return counter.most_common(1)[0][0]
        else:
            return np.mean(y)