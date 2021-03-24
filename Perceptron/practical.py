from random import randint, seed
from perceptron import PerceptronTrain, PerceptronTest
from structures import Vector, Scalar, Dataset
from utils import train_test_split
seed(42)

# generate data
v = Vector(randint(-100, 100), randint(-100, 100))
xs = [Vector(randint(-100, 100), randint(-100, 100)) for i in range(500)]
ys = [v * x * Scalar(randint(-1, 9)) for x in xs]
# generate vector-label pairs
data = [(vector, label) for vector, label in zip(xs, ys)]
# split data
train, test = train_test_split(data)

weights, bias = PerceptronTrain(train)

print(v * weights)
print(v.magnitude()*weights.magnitude())
print(bias)
# res = PerceptronTest(weights, bias, data)

# print(res)
