class Node:
    def __init__(self, split_variable, split_type, split_value, is_leaf = False, pred = None):
        self.left = None
        self.right = None
        self.split_variable = split_variable 
        self.split_type = split_type 
        self.split_value = split_value
        self.is_leaf = is_leaf
        self.pred = pred

    def __eq__(self, other):
        if isinstance(other, Node):
            c1 = self.split_variable == other.split_variable
            c2 = self.split_type == other.split_type
            c3 = self.split_value == other.split_value
            c4 = self.is_leaf == other.is_leaf  
            c5 = self.pred == other.pred
            return c1 and c2 and c3 and c4 and c5       
        return NotImplemented
    

    def printTree(self, level=0):
        if self != None:
            if self.is_leaf:
                print(' ' * 4 * level,  '--', self.pred)
            else:
                self.left.printTree(level + 1)
                print(' ' * 4 * level,  '--', self.split_variable, self.split_type, self.split_value)
                self.right.printTree(level + 1)
