import sys
from Tree import Tree
from random import shuffle

class DecisionTreeTrain():

    def build_tree(self, data, features_left, maxdepth=2):
        guess = self.guess(data) # get the most common label

        if self.is_unambiguous(data, 'ok'):
            return guess

        elif len(features_left) == 0:
            return guess

        if maxdepth < 2:
            return guess

        best = self.top_feature(data, 'ok', features_left)

        # necessary if the potential maximum depth of the tree is unknown
        if self.is_unambiguous(data, best):
            return guess

        features_left = [feat for feat in features_left if feat != best]

        return Tree(data='is_%s' % (best),
                    left=self.build_tree(data[data[best] == False],
                                         features_left,
                                         (maxdepth-1)),
                    right=self.build_tree(data[data[best] == True],
                                          features_left,
                                          (maxdepth-1)))


    def guess(self, data) -> Tree:
        """ returns a most frequent value in a current dataset as a Leaf """
        return Tree.leaf(data.ok.value_counts().idxmax())


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


    def top_feature(self, data, goal, features,  func=max) -> str:
        """
            returns the feature with the highest (by default) or lowest score
        """
        return func(
            features, key=lambda f: self.single_feature_score(data, goal, f)
            )
