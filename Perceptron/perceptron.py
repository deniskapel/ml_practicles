from structures import Scalar, Vector, Dataset
from typing import Union, List


def PerceptronTrain(data: Dataset, Maxiter: int = 10):
    """ returns the weights for the perceptron """

    # initiate essentials
    bias = Scalar(0)
    weights = Vector.zero(len(data[0][0]))

    for i in range(Maxiter):
        # process the first vector
        for d in data:
            vec = d[0]
            label = d[1]
            # compute activation
            activation = vec * weights + bias
            # check if label and activation have different signs
            if (activation * label).sign() <= 0:
                weights = weights + label*vec
                bias += label

    return weights, bias


def PerceptronTest(weights: Vector, bias: Scalar, data: Dataset):
    """ 
        takes weights, bias and vectors 
        and returns their labels
    """
    res = []
    for d in data:
        # compute activate
        vec = d[0]
        activation = vec * weights + bias
        res.append(activation.sign())

    return res
