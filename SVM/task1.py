import scipy.optimize
import numpy.random


def f(x):
    return x**2


def df(x):
    return 2*x


n = numpy.random.randint(-1000, 1000)
print(n, '\n')

print(scipy.optimize.minimize(f, n, jac=df).x)
print('\n')
print(scipy.optimize.minimize(f, n, jac=False).x)
