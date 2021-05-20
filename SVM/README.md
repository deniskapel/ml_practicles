# Support vector machines practical

## 1. Read the documentation for scipy.optimize.minimize, paying special attention to the Jacobian argument jac. Who computes the gradient, the minimize function itself, or the developer using it?

The gradient is computed by the derivative passed to the function through a jac argument. If jac is `False`, it is estimated using two-point difference.

d and df functions are in [test_scipy.py](test_scipy.py). The first output is with_gradient and the second is without it.

```python
    python3 test_scipy.py
    -199 

        fun: 0.0
    hess_inv: array([[0.5]])
        jac: array([0.])
    message: 'Optimization terminated successfully.'
        nfev: 7
        nit: 4
        njev: 7
    status: 0
    success: True
            x: array([0.])


        fun: 5.54950786776595e-17
    hess_inv: array([[0.49999997]])
        jac: array([2.1573778e-12])
    message: 'Optimization terminated successfully.'
        nfev: 16
        nit: 5
        njev: 8
    status: 0
    success: True
            x: array([-7.44950191e-09])
```

Both functions terminate optimization successfully, though `with_gradient` computes it faster (nit: 4<5) and the result is more accurate (x = 0 and ~0).


## Tasks 2-4. 

The function `svm_loss`, `svm` and `gradient_svm_loss` is in [here](utils.py).

## 5. Graph the three hyperplanes found by training: an averaged perceptron, support vector machine without gradient and support vector machine with gradient.

Two isolated clusters are generated in [here](dataset.py). [An averaged perceptron](AveragedPerceptron.py) is taken from the previous practical, though it was modified to use numpy instead of self-design data stractures.

The [graph](svm-svm-perceptron.pdf) demonstrates the result of three approaches. All three split the data properly, though averaged perceptron is less stable in terms of its padding. SVM and SVM_with_gradient show very similar results (same if initial bias and weights are hard coded) but the gradient function helps the algorithm converge faster.