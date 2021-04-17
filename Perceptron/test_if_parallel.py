from random import randint, seed
from perceptron import PerceptronTrain, PerceptronTest
from structures import Vector, Scalar, Dataset
from utils import train_test_split
seed(42)

scores = []
# generate data
for i in range(10000):
    v = Vector(randint(-100, 100), randint(-100, 100))
    xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
    ys = [v * x * Scalar(randint(-1, 9)) for x in xs]
    # generate vector-label pairs
    data = [(vector, label) for vector, label in zip(xs, ys)]
    # split data
    train, test = train_test_split(data)
    weights, bias = PerceptronTrain(train)

    scores.append(
        ((v*weights) / (v.magnitude()*weights.magnitude())).val
    )

print(
    "average (v*w)/Scalar(v.magnitude() * w.magnitude()) is %f" % (sum(scores) / len(scores))
)
