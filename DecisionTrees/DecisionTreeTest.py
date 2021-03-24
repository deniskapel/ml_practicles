from Tree import Tree
from DecisionTreeTrain import DecisionTreeTrain

class DecisionTreeTest():

    def __init__(self, tree, test_point):
        self.tree = tree
        self.test_point = test_point # pandas row

    def predict(self):
        """ predicts an ok value based on given features """
        if self.tree.is_leaf():
            return self.tree.data

        if self.test_point[self.tree.data[3:]].values[0] == False:
            left = DecisionTreeTest(self.tree.left, self.test_point)
            return left.predict()

        elif self.test_point[self.tree.data[3:]].values[0] == True:
            right = DecisionTreeTest(self.tree.right, self.test_point)
            return right.predict()
