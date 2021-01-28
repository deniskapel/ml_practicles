# These are notes for some of the tasks

## Task 3 ##
The best feature for the whole dataset is **systems**. Our prediction is correct 90% of the time.

The worst one is **easy**. Our prediction is correct 60% of the time


## Task 4 ##
The trained tree performs 5% better than a single feature classifier
(the same dataset was used for zero/one loss function).

**NB**: after we split the dataset using **systems**, and follow the right branch
we arrive at the point when all features have the same score,
and as **easy** goes first in the list,
the best (and the worst) role is assigned to it.
The solution might be to shuffle features every time

**Performance**
loss = 1/20 for model vs loss = 2/20 max for single features

## Task 5 ##
The performance stops rising after level 4, so no point to train a larger model.

**Performance**
loss = 1/20 for depth >= 4 levels vs loss = 2/20 max for depth < 4
