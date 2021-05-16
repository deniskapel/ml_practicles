import numpy
from random import seed

numpy.random.seed(42)

x_plus = numpy.random.normal(loc=[-1, -1], scale=0.5, size=(20, 2))
x_minus = numpy.random.normal(loc=[1, 1], scale=0.5, size=(20, 2))

X = []

for x in x_plus:
    X.append((x, 1))

for x in x_minus:
    X.append((x, -1))
