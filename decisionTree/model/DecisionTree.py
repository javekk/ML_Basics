import numpy as np
import itertools

from model.Node import Node


class DecisionTree:
    def __init__(self):
        self.model = Node(None, None, None)


    def is_categorical(self, d):
        return d.dtypes.name == 'category'


    def gini_impurity(y):
        P = y.value_counts() / y.shape[0]
        return 1 - np.sum(P**2)


    def entropy(y):
        P = y.value_counts() / y.shape[0]
        return - np.sum(P * np.log2(P + 1e-20))


    def variance(self, y):
        return y.var() if (len(y) != 1) else 0


    def information_gain(self, y, mask, func = entropy):   
        no_true = sum(mask) 
        no_false = len(mask) - no_true 
        perc_true = no_true / (no_true + no_false)
        perc_false = no_false / (no_true + no_false)

        if self.is_categorical(y):
            return func(y) - perc_true * func(y[mask]) - perc_false * func(y[-mask])
        else:
            return self.variance(y) - (perc_true * self.variance(y[mask])) - (perc_false * self.variance(y[-mask]))    


    def categorical_options(self, y):
        no_classes = y.unique()
        split_poss = []
        for i_class in range(0, len(no_classes)+1):
            for subset in itertools.combinations(no_classes, i_class):
                split_poss.append(list(subset))
        # Remove first and last because the split will include all values
        return split_poss[1:-1]


    def max_information_gain_split(self, feature, y, func=entropy):
        is_numerical = False if self.is_categorical(feature) else True
        
        if is_numerical:
            options = feature.sort_values().unique()[1:-1]
        else: 
            options = self.categorical_options(feature)

        if len(options) == 0:
            return (None, None, None, False)

        split_values = []
        info_gains = [] 
        best_info_gain = 0
        best_split = []

        for s in options:
            mask = feature < s if is_numerical else feature.isin(s)
            info_gain = self.information_gain(y, mask, func)
            info_gains.append(info_gain)
            split_values.append(s)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_split = s

        return (best_info_gain, best_split, is_numerical, True)


    def get_best_split(self, X, y):
        masks = X.apply(self.max_information_gain_split, y = y).T
        masks.columns = ['info_gain', 'split', 'is_numerical', 'is_usable_as_split' ]

        if sum(masks.is_usable_as_split) == 0: # Check if we have no splits left
            return(None, None, None, None)
        else:
            masks = masks.loc[masks.is_usable_as_split]
            split_index = masks.info_gain.astype('float').argmax()
            split_value = masks.iloc[split_index, 1] 
            info_gain = masks.iloc[split_index, 0]
            is_numeric = masks.iloc[split_index, 2]
            split_variable = masks.index[split_index]
            return(split_variable, split_value, info_gain, is_numeric)


    def make_split(self, data, split_variable, split_value, is_numeric):
        if is_numeric:
            mask = data[split_variable] < split_value
            left = data[mask]
            right = data[-mask]
        else:
            mask = data[split_variable].isin(split_value)
            left = data[mask]
            right = data[-mask]
        return(left,right)


    def leaf_output(self, y_cluster, is_numeric):
        return y_cluster.mean() if is_numeric else y_cluster.value_counts().idxmax()


    def check_max_category(self, data, max_categories):
        check_columns = data.dtypes[data.dtypes == "category"]
        for column in check_columns:
            var_length = len(data.loc[column].unique()) 
            if var_length > max_categories:
                raise ValueError('Column ' + column + ' has '+ str(var_length) + ' unique values, max = ' +  str(max_categories))


    def _train_tree(
        self, 
        X, 
        y, 
        is_y_numeric, 
        max_depth = None, 
        min_samples_split = None, 
        min_info_gain = 1e-10, 
        depth = 0, 
        max_categories = 20
    ):
        if depth == 0:
            self.check_max_category(X, max_categories)

        depth_cond = max_depth == None or depth < max_depth
        sample_cond = min_samples_split == None or X.shape[0] > min_samples_split

        if depth_cond and sample_cond:
            split_variable, split_value, info_gain, is_feature_numeric = self.get_best_split(X, y)
            
            if (info_gain is not None and info_gain >= min_info_gain):
                left, right = self.make_split(X, split_variable, split_value, is_feature_numeric)
                split_type = "<=" if is_feature_numeric else "in"
                tree = Node(split_variable, split_type, split_value)
                depth += 1
                l_tree = self._train_tree(left, y.loc[left.index], is_y_numeric, max_depth, min_samples_split, min_info_gain, depth)
                r_tree = self._train_tree(right, y.loc[right.index], is_y_numeric, max_depth, min_samples_split, min_info_gain, depth)
                # if l == r:
                #   prune()
                tree.left = l_tree
                tree.right = r_tree
                return tree

            else:
                pred = self.leaf_output(y, is_y_numeric)
                return Node(None, None, None, True, pred)

        else:
            pred = self.leaf_output(y, is_y_numeric)
            return Node(None, None, None, True, pred)


    def fit(
        self, 
        X, 
        y, 
        is_y_numeric, 
        max_depth = None, 
        min_samples_split = None, 
        min_info_gain = 1e-10, 
        depth = 0, 
        max_categories = 20
    ):
        self.model = self._train_tree(X, y, is_y_numeric, max_depth, min_samples_split, min_info_gain, depth, max_categories)


    def _predict(self, x_i, node):
        if node.is_leaf:
            return node.pred
        elif node.split_type == "<=": #is_numeric
            if x_i[node.split_variable] <= node.split_value:
                return self._predict(x_i, node.left)
            else:
                return self._predict(x_i, node.right)
        else: 
            if x_i[node.split_variable] in node.split_value:
                return self._predict(x_i, node.left)
            else:
                return self._predict(x_i, node.right)


    def predict(self, x_i):
        return self._predict(x_i, self.model)