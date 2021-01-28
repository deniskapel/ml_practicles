import pandas as pd
from Tree import Tree
from DecisionTreeTrain import DecisionTreeTrain
from DecisionTreeTest import DecisionTreeTest

def zero_one_loss(tree, test, labels):
    loss = 0
    for i, label in enumerate(labels):
        if label != DecisionTreeTest(tree, test.iloc[[i]]).predict():
            loss += 1

    print(loss)

"""
    Task 1
"""
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

print(tree, '\n')

"""
    Task 2
"""
df = pd.read_csv('data.csv')
df = df.assign(ok=lambda df: df.rating >= 0)
print(df, '\n')


"""
    Task 3
"""
features = list(df.columns[1:-1])
train = DecisionTreeTrain(df)

print(train.best_feature(train.data, 'ok', features),
      ' is the best\n')
print(train.worst_feature(train.data, 'ok', features),
      ' is the worst\n')

"""
  Task 4
"""
single_feature_trees = [DecisionTreeTrain(df).build_tree([feature]) for feature in features]
decision_tree = train.build_tree(features, 4)
labels = list(df.ok)
print(decision_tree, '\n')


print('\nThis is the performance comparison: single features vs overall (last)')
[zero_one_loss(tree, df, labels) for tree in (single_feature_trees + [decision_tree])]


"""
    Task 5
"""
print('\nThis is the tree performance comparison: from single row to max')
diff_depth = [DecisionTreeTrain(df).build_tree(features, depth) for depth in range(2,8)]
[zero_one_loss(tree, df, labels) for tree in diff_depth]
