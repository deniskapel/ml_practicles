from structures import Dataset
from random import sample


def train_test_split(data: Dataset, test_size=0.1):
    """ split data into 90/10% sets """
    test = sample(data, int(len(data) * test_size))
    train = list(set(data) - set(test))

    return train, test
