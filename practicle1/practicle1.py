import pandas as pd
from Tree import Tree

# Task 1
l1 = Tree.leaf('like')
l2 = Tree.leaf('nah')

tree = Tree(data='isSystem',
            left=l1,
            right=Tree(data="takenOtherSys?",
                       left=Tree(data='morning?',
                                 left=l1,
                                 right=l2),
                       right=Tree(data='likedOtherSys?',
                                  left=l2,
                                  right=l1)))

# Task 2
df = pd.read_csv('data.csv')
df = df.assign(ok=lambda df: df.rating >= 0)

# Task 3 and 4

features = list(df.columns[1:-1])

class DecisionTreeTrain():

    def __init__(self, data, features):
        self.data = data # pandas DataFrame
        self.features_left = features # list of strings

    def decide(self, subset):
        guess = self.guess(self, subset)
        if is_unambigues(self, subset):
            return quess

        elif len(self.features_left) == 0:
            return guess

        else:
            feat = self.best_feature(self.data, 'ok', self.features_left)
            self.reduce_features(feat)

            left = self.data[self.data[feature == False]]
            right = self.data[self.data[feature == True]]

            left = DecisionTreeTrain(left, self.features_left)
            right = DecisionTreeTrain(right, self.features_Left)


    def guess(self, subset) -> str:
        """ return a most frequent value in a current dataset """
        return Tree.leaf(
            subset.ok.value_counts().idxmax()
            )

    def is_unambigous(self, subset, feature) -> bool:
        """ check if there are no other options """
        return self.single_feature_score(subset, subset.ok, feature) == 1

    def reduce_features(self, feature):
        """ removes a given feature from self.features_left """
        self.features_left.remove(feature)

    def single_feature_score(self, data, goal, feature) -> float:
        """ calculates  a ratio of correct / total answers """
        return len(data[data[feature] == data[goal]]) / len(data)

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


train = DecisionTreeTrain(df, features)
print(train.best_feature(train.data, 'ok', train.features_left))
print(train.worst_feature(train.data, 'ok', train.features_left))
# best is ai, worst is systems

print(train.guess(df))
