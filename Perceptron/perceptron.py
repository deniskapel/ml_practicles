from structures import Scalar, Vector, Dataset
from typing import Union, List
from random import sample, shuffle


def PerceptronTrain(data: Dataset, Maxiter: int = 10, shuffling=False):
    """ 
        returns the weights for the perceptron 
        if shuffling is set to True, 
        the data is randomly permutated at each of the epochs 
    """
    # initiate essentials
    bias = Scalar(0)
    weights = Vector.zero(len(data[0][0]))
    for i in range(Maxiter):
        if shuffling:
            # sample is used no to mofidy the original data
            data = sample(data, len(data))

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


def AveragedPerceptronTrain(data: Dataset, Maxiter: int = 10, shuffling=False):
    """ 
        returns the weights for the perceptron 
        if shuffling is set to True, 
        the data is randomly permutated at each of the epochs 
    """
    # initiate essentials
    bias = Scalar(0)
    weights = Vector.zero(len(data[0][0]))
    cached_bias = Scalar(0)
    cached_weights = Vector.zero(len(data[0][0]))
    counter = Scalar(0)
    for i in range(Maxiter):
        if shuffling:
            # sample is used no to mofidy the original data
            data = sample(data, len(data))

        # process the first vector
        for d in data:
            vec = d[0]
            label = d[1]
            # compute activation
            activation = vec * weights + bias
            # check if label and activation have different signs
            if (activation * label).sign() <= 0:
                # update weights
                weights = weights + label*vec
                # update bias
                bias += label
                # update cached weights
                cached_weights = cached_weights + label * counter * vec
                cached_bias = cached_bias + label * counter
            # increment counter when go to the next vector
            counter += Scalar(1)

    bias = bias - (Scalar(1)/counter) * cached_bias
    weights = weights - (Scalar(1) / counter) * cached_weights

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
