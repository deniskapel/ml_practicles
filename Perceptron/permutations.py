import matplotlib.pyplot as plt
from random import randint, seed, sample
from perceptron import PerceptronTrain, PerceptronTest
from structures import Vector, Scalar, Dataset
from utils import train_test_split, eval_predictions
seed(42)

v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]

# generate vector-label pairs
data = [(vector, label) for vector, label in zip(xs, ys)]
# split data
train, test = train_test_split(data)

# retrieve labels only from a test set
y_true = [entry[1].sign() for entry in test]

sorted_train = sorted(train, key=lambda vec: vec[1].val)

no_permutation = []
init_permutation = []
permutation_by_epoch = []

for maxiter in range(1, 100, 5):
    # no permutation
    weights, bias = PerceptronTrain(sorted_train, Maxiter=maxiter)
    preds = PerceptronTest(weights, bias, test)
    no_permutation.append(eval_predictions(preds, y_true))
    # only initial permutation - sample does not change the original list
    shuffled = sample(sorted_train, len(sorted_train))
    weights, bias = PerceptronTrain(shuffled, Maxiter=maxiter)
    preds = PerceptronTest(weights, bias, test)
    init_permutation.append(eval_predictions(preds, y_true))
    # permutate by each epoch
    weights, bias = PerceptronTrain(
        sorted_train, Maxiter=maxiter, shuffling=True)
    preds = PerceptronTest(weights, bias, test)
    permutation_by_epoch.append(eval_predictions(preds, y_true))

x = [x for x in range(1, 100, 5)]

plt.plot(x, no_permutation,
         color='blue', label='no permutation', marker='o')
plt.plot(x, init_permutation, color='red',
         label='random permutation at the beginning', marker='o')
plt.plot(x, permutation_by_epoch,
         color='green', label='random permutation at each epoch', marker='o')

plt.xlabel('Number of epochs to train', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend()
plt.show()
