# Support vector machines practical

## 1. Read the documentation for scipy.optimize.minimize, paying special attention to the Jacobian argument jac. Who computes the gradient, the minimize function itself, or the developer using it?

## Tasks 2-4. 

The function `svm_loss`, `svm` and `gradient_svm_loss` is in [here](utils.py).

## 5. Graph the three hyperplanes found by training: an averaged perceptron, support vector machine without gradient and support vector machine with gradient.

Two isolated clusters are generated in [here](dataset.py). [An averaged perceptron](AveragedPerceptron.py) is taken from the previous practical, though it was modified to use numpy instead of self-design data stractures.

The [graph](svm-svm-perceptron.pdf) demonstrates the result of three approaches. All three split the data properly, though averaged perceptron is less stable in terms of its padding. SVM and SVM_with_gradient show very similar results (same if initial bias and weights are hard coded) but the gradient function helps the algorithm converge faster.