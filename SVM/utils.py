import numpy
from scipy.optimize import minimize


def hinge_loss_surrogate(y_gold: int, y_pred: int) -> int:
    """ hinge loss implementation """
    return max(0, 1 - y_gold * y_pred)


def svm_loss(weights_and_bias: float, D: list, C: int = 1):
    """
        A support vector machine loss function.
        params:
        weights_and_bias: list of floats -
            1-D list - a scipy constraint for optimize.minimize
        D: list of tuples - each tuple is a pair of
            numpy.array(features) and a label (float or int)
        C: int - number of points to move across a hyperplane to a proper side
            1 - is a default

        it uses the hinge_loss_surrogate function to calculate loss
    """
    # replace with your implementation, must call hinge_loss_surrogate
    w = weights_and_bias[0:2]
    b = weights_and_bias[2]

    large_margin = 0.5 * numpy.dot(w, w)

    loss = 0
    for pair in D:
        # count hinge loss for each pair
        loss += hinge_loss_surrogate(
            # y_true
            pair[1],
            # y_pred
            numpy.dot(pair[0], w) + b)

    small_slack = C * loss

    return large_margin + small_slack


def svm(f_to_minimize, D, C=1, use_gradient=False):
    """
        a function to compute optimized weights and bias,
        using the svm_loss function
    """
    x0 = numpy.random.rand(3)  # x0[0:2] weights, x0[2] bias
    res = minimize(f_to_minimize, x0, args=(D, C), jac=use_gradient).x
    return res[0:2], res[2]


def gradient_hinge_loss_surrogate(y_gold, y_pred):
    if hinge_loss_surrogate(y_gold, y_pred) == 0:
        # return [0, 0]
        return 0
    else:
        # return [-y_pred, -y_gold]
        return -y_gold


def gradient_svm_loss(weights_and_bias: list, D, C=1):
    # implement the gradient of svm_loss
    w = weights_and_bias[0:2]
    b = weights_and_bias[2]

    d_w = w + C*sum(
        [pair[0]*gradient_hinge_loss_surrogate(
            pair[1], numpy.dot(pair[0], w) + b) for pair in D])

    d_b = C*sum(
        [gradient_hinge_loss_surrogate(
            pair[1], numpy.dot(pair[0], w) + b) for pair in D])

    # open up weights to return 1-D array: a scipy constraint
    return [*d_w, d_b]


def hyperplane(weights, b, xmin, xmax):
    """
        recreate hyperplane from weights using Chris Albon's recipe
        https://chrisalbon.com/machine_learning/support_vector_machines/plot_support_vector_classifier_hyperplane/
    """
    a = -weights[0] / weights[1]
    xx = numpy.linspace(xmin, xmax)
    yy = a * xx - b/weights[1]

    return xx, yy
