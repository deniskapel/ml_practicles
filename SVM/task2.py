from utils import svm, svm_loss
from dataset import X, x_plus

weights, bias = svm(svm_loss, X)

print(weights, bias)

print(
    svm_loss([-1.35964324, -2.28782899, 0.0175], X)
)

print(
    svm_loss([-0.99026836, -1.22045837, 0.039363277504995256], X)
)
