from structures import Scalar, Vector

x = Scalar(4.0)
y = Scalar(2.0)
a = Vector(1.0, 2.0, 3.0)
b = Vector(3.0, 4.0, 5.0)
# __add__
print(x+y)
# __sub__
print(x - y)
# __mul__ Scalar
print(x * y)
# __mul__ Vec
print(y * a)


# __truediv__
print(x / y)
# __rtruediv__
print(y.__rtruediv__(b))


# sign
print(x.sign())
print(Scalar(-7).sign())
print(Scalar(0).sign())


# Vector + Vector
print(a + b)
# magnitute
print(Vector(1, 1, 1, 1).magnitude())
# multiplication
print(a * b)


print(Vector.zero(3))

# Errors handling
# Vectors
# print(a + 3)
# print(a+Vector(1, 2))

# Scalars
# __mul__ WrongType
# print(x*dict())
