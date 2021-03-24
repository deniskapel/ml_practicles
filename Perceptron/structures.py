from typing import Union, List
from math import sqrt


Dataset = List[tuple]


class Scalar:
    pass


class Vector:
    pass


class Scalar:
    def __init__(self: Scalar, val: float):
        self.val = float(val)

    def __mul__(self: Scalar, other: Union[Scalar, Vector]) -> Union[Scalar, Vector]:
        # hint: use isinstance to decide what `other` is
        # raise an error if `other` isn't Scalar or Vector!
        if isinstance(other, Scalar):
            return Scalar(self.val * other.val)
        if isinstance(other, Vector):
            return Vector(*[self.val * x for x in other.entries])
        else:
            raise TypeError

    def __add__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val + other.val)

    def __sub__(self: Scalar, other: Scalar) -> Scalar:
        return Scalar(self.val - other.val)

    def __truediv__(self: Scalar, other: Scalar) -> Scalar:
        # implement division of scalars
        if isinstance(other, Scalar):
            return Scalar(self.val / other.val)
        else:
            raise TypeError

    def __rtruediv__(self: Scalar, other: Vector) -> Vector:
        # implement division of vector by scalar
        if isinstance(other, Vector):
            return Vector(*[x * (float(1) / self.val) for x in other.entries])
        else:
            raise TypeError

    def __repr__(self: Scalar) -> str:
        return "Scalar(%r)" % self.val

    def sign(self: Scalar) -> int:
        # returns -1, 0, or 1
        if self.val > 0:
            return 1
        elif self.val < 0:
            return -1
        else:
            return 0

    def __float__(self: Scalar) -> float:
        return self.val


class Vector:
    def __init__(self: Vector, *entries: List[float]):
        self.entries = entries

    def zero(size: int) -> Vector:
        return Vector(*[0 for i in range(size)])

    def __add__(self: Vector, other: Vector) -> Vector:
        if not isinstance(other, Vector):
            raise TypeError
        if not self.__len__() == other.__len__():
            raise ValueError
        return Vector(*[x+y for x, y in zip(self.entries, other.entries)])

    def __mul__(self: Vector, other: Vector) -> Scalar:
        if not isinstance(other, Vector):
            raise TypeError
        if not self.__len__() == other.__len__():
            raise ValueError
        dot = 0
        for x, y in zip(self.entries, other.entries):
            dot += x*y
        return Scalar(dot)

    def magnitude(self: Vector) -> Scalar:
        sum_of_squares = 0
        for x in self.entries:
            sum_of_squares += x*x
        return Scalar(sqrt(sum_of_squares))

    def unit(self: Vector) -> Vector:
        return self / self.magnitude()

    def __len__(self: Vector) -> int:
        return len(self.entries)

    def __repr__(self: Vector) -> str:
        return "Vector%s" % repr(self.entries)

    def __iter__(self: Vector):
        return iter(self.entries)
