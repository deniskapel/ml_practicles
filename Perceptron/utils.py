from structures import Dataset
from random import sample


def train_test_split(data: Dataset, test_size=0.1):
    """ split data into 90/10% sets """
    # take a sample of text_size
    test = sample(data, int(len(data) * test_size))
    # separate a test sample from an original set and store it a trainset
    train = list(set(data) - set(test))

    return train, test


def eval_predictions(preds: list, y_true: list):
    """
        compares predicated and true labels
        return the percentage of predicted correctly
    """
    count_true = 0
    for pred, true in zip(preds, y_true):
        if pred == true:
            count_true += 1

    return count_true / len(y_true)
