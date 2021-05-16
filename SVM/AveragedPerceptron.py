import numpy
from random import sample


def AveragedPerceptronTrain(data: list, Maxiter: int = 10, shuffling=False):
    """ 
        returns the weights for the perceptron 
        if shuffling is set to True, 
        the data is randomly permutated at each of the epochs 
    """
    # initiate essentials
    bias = numpy.float64(0)
    weights = numpy.zeros(len(data[0][0]))
    cached_bias = numpy.float64(0)
    cached_weights = numpy.zeros(len(data[0][0]))
    counter = numpy.uint(0)

    for i in range(Maxiter):
        if shuffling:
            # sample is used no to mofidy the original data
            data = sample(data, len(data))

        # process the first vector
        for d in data:
            vec = d[0]
            label = d[1]
            # compute activation
            activation = numpy.dot(vec, weights) + bias
            # check if label and activation have different signs
            if numpy.sign(activation * label) <= 0:
                # update weights
                weights = weights + label*vec
                # update bias
                bias += label
                # update cached weights
                cached_weights = cached_weights + label * counter * vec
                cached_bias = cached_bias + label * counter
            # increment counter when go to the next vector
            counter += numpy.uint(1)

    bias = bias - (numpy.uint(1)/counter) * cached_bias
    weights = weights - (numpy.uint(1) / counter) * cached_weights

    return weights, bias


def PerceptronTest(weights: numpy.array, bias: numpy.float64, data: list):
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
