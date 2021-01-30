import pandas as pd
from Tree import Tree
from DecisionTreeTrain import DecisionTreeTrain
from DecisionTreeTest import DecisionTreeTest

def zero_one_loss(tree, test_set, labels):
    loss = 0
    for i, label in enumerate(labels):
        if label != DecisionTreeTest(tree, test_set.iloc[[i]]).predict():
            loss += 1

    return loss

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
train = DecisionTreeTrain()

print(train.top_feature(df,'ok', features, max),
      ' is the best\n')
print(train.top_feature(df, 'ok', features, min),
      ' is the worst\n')

"""
  Task 4
"""
single_feature_trees = [train.build_tree(df, [feature]) for feature in features]
decision_tree = train.build_tree(df, features, 4)
labels = list(df.ok)
print(decision_tree, '\n')
print("Tree's depth is ", decision_tree.depth(), '\n')



print('\nThe performance comparison: single features vs overall (last)')
print(
      [zero_one_loss(tree, df, labels) for tree in (single_feature_trees + [decision_tree])]
      )


"""
    Task 5
"""
print('\nThe performance comparison: from random_balanced_choice to to max.\n')
diff_depth = [train.build_tree(df, features, depth) for depth in range(1,7)]
print(
      [zero_one_loss(tree, df, labels) for tree in diff_depth]
      )
