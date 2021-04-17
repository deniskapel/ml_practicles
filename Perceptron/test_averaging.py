from utils import train_test_split, eval_predictions
from structures import Vector, Scalar, Dataset
from perceptron import PerceptronTrain, PerceptronTest, AveragedPerceptronTrain
from random import randint, seed, sample
import matplotlib.pyplot as plt
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

regular = []
regular_shuffled = []
averaged = []
averaged_shuffled = []

for maxiter in range(1, 100, 5):
    # regular perceptron, no shuffling at each epoch
    weights, bias = PerceptronTrain(train, Maxiter=maxiter, shuffling=False)
    preds = PerceptronTest(weights, bias, test)
    regular.append(eval_predictions(preds, y_true))
    # regular perceptron with shuffling at each epoch
    weights, bias = PerceptronTrain(train, Maxiter=maxiter, shuffling=True)
    preds = PerceptronTest(weights, bias, test)
    regular_shuffled.append(eval_predictions(preds, y_true))
    # averaged perceptron, no shuffling at each epoch
    weights, bias = AveragedPerceptronTrain(train, Maxiter=maxiter)
    preds = PerceptronTest(weights, bias, test)
    averaged.append(eval_predictions(preds, y_true))
    # averaged perceptron with shuffling at each epoch
    weights, bias = AveragedPerceptronTrain(
        train, Maxiter=maxiter, shuffling=True)
    preds = PerceptronTest(weights, bias, test)
    averaged_shuffled.append(eval_predictions(preds, y_true))

x = [x for x in range(1, 100, 5)]

plt.plot(x, regular,
         color='blue', label='regular', marker='o')
plt.plot(x, regular_shuffled, color='red',
         label='regular shuffled at each epoch', marker='o')
plt.plot(x, averaged,
         color='green', label='averaged', marker='o')
plt.plot(x, averaged_shuffled,
         color='grey', label='averaged shuffled at each epoch', marker='o')
plt.xlabel('Number of epochs to train', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend()
plt.show()
