from dataset import x_plus, x_minus, X
from utils import svm, svm_loss, gradient_svm_loss, hyperplane
import matplotlib.pyplot as plt
from AveragedPerceptron import AveragedPerceptronTrain

xlim, ylim = -3, 2

# Averaged perceptron hyperplane
weights, bias = AveragedPerceptronTrain(X, shuffling=True)
xx, yy = hyperplane(weights, bias, xlim, ylim)
plt.plot(xx, yy, color='green', label='AveragePerceptronTrain')

# support vector machine without gradient
weights, bias = svm(svm_loss, X)
xx, yy = hyperplane(weights, bias, xlim, ylim)
plt.plot(xx, yy, color='grey', label='SVM without gradient')

# support vector machine with gradient
weights, bias = svm(svm_loss, X, use_gradient=gradient_svm_loss)
xx, yy = hyperplane(weights, bias, xlim, ylim)
plt.plot(xx, yy, color='brown', label='SVM with gradient')


plt.scatter(
    x_plus[:, 0], x_plus[:, 1],
    marker='+',
    color='blue'
)

plt.scatter(
    x_minus[:, 0], x_minus[:, 1],
    marker='x',
    color='red'
)


plt.xlabel('x-axis', fontsize=12)
plt.ylabel('y-axis', fontsize=12)
plt.legend()
# plt.show()

plt.savefig("svm-svm-perceptron.pdf")
