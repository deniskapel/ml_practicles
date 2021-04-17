from random import randint, seed
from perceptron import PerceptronTrain, PerceptronTest
from structures import Vector, Scalar, Dataset
from utils import train_test_split, eval_predictions
seed(42)

xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [Scalar(1) if x.entries[0] * x.entries[1] < 0 else Scalar(-1) for x in xs]

# generate vector-label pairs
data = [(vector, label) for vector, label in zip(xs, ys)]

# split data
train, test = train_test_split(data)
# retrieve labels only from a test set
y_true = [entry[1].sign() for entry in test]

# train
weights, bias = PerceptronTrain(train)

# apply perceptron to classify test set
preds = PerceptronTest(weights, bias, test)

# evaluate the performance
score = eval_predictions(preds, y_true)

print(score)
