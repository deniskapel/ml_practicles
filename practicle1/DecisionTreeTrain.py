import sys
from Tree import Tree
from random import shuffle

class DecisionTreeTrain():

    def __init__(self, data):
        self.data = data

    def build_tree(self, features, maxdepth=2):
        guess = self.guess(self.data) # get the most common label

        if self.is_unambiguous(self.data, 'ok'):
            return guess

        elif len(features) == 0:
            return guess

        else:
            if maxdepth < 2:
                return guess

            feat = self.best_feature(self.data, 'ok', features)

            if self.is_unambiguous(self.data, feat):
                # first check that the data in the featured column is not uniform
                return guess

            remaining_features = [feature for feature in features if feature != feat]

            right = DecisionTreeTrain(self.data[self.data[feat] == True])

            left = DecisionTreeTrain(self.data[self.data[feat] == False])

            return Tree(data='is_%s' % (feat),
                        left=left.build_tree(remaining_features,
                                             (maxdepth-1)),
                        right=right.build_tree(remaining_features,
                                               (maxdepth-1)))

    def guess(self, data) -> Tree:
        """ returns a most frequent value in a current dataset as a Leaf """
        if len(data) > 0:
            return Tree.leaf(data.ok.value_counts().idxmax())

        return None


    def is_unambiguous(self, data, feature) -> bool:
        """ check if there are no other options"""
        return len(data[feature].value_counts().keys()) == 1

    def single_feature_score(self, data, goal, feature) -> float:
        """ calculates a ratio of correct / total answers """
        left = max( # counts the most probable answer if feature is False
            data[data[feature] == False][goal].value_counts().values,
            default=0)
        right = max( # counts the most probable answer if feature is True
            data[data[feature] == True][goal].value_counts().values,
            default=0)

        return (left + right) / len(data)

    def best_feature(self, data, goal, features) -> str:
        """ returns the feature with the highest score"""
        return max(
            features, key=lambda f: self.single_feature_score(data, goal, f)
            )

    def worst_feature(self,data, goal, features) -> str:
        """ returns a feature with the lowest score """
        return min(
            features, key=lambda f: self.single_feature_score(data, goal, f)
            )
