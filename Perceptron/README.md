# Perceptron practical

## Task 1 Implement your own Scalar and Vector classes, without using any other modules
```python3
    run basic_operations.py
```
The [files](basic_operations.py) prints out results of all basic operations for two classes

## Task 2 Implement the PerceptronTrain and PerceptronTest functions, using your Vector and Scalar classes. Do not permute the dataset when training; run through it linearly.

Both functions are in [perceptron.py](perceptron.py).

## Task 3 Make a 90-10 test-train split and evaluate your algorithm on the following dataset

Vectros seem to be parallel.

```python3
    $ python3 test_if_parallel.py 
    average (v*w)/Scalar(v.magnitude() * w.magnitude()) is 0.928984
```

## Task 4 Make a 90-10 test-train split and evaluate your algorithm on the xor dataset:

Perceptron classifies about 50% of the data correctly.

```python3
    $ python3 xor_performance.py 
    0.46
```

## Task 5 Sort the training data from task 3 so that all samples with y < 0 come first, then all samples with y = 0, then all samples with y > 0. (That is, sort by y.)

### Graph the performance (computed by PerceptronTest) on both train and test sets versus epochs for perceptrons trained on no permutation, random permutation at the beginning, random permutation at each epoch

[Graph](permutations.png)

Shuffling at each epoch shows higher performance though it can get overfit big from time to time. More or less nice performance of `no_shuffling` is due to the fact that the task is relatevily easy.

## Implement AveragedPerceptronTrain; using random permutation at each epoch, compare its performance with PerceptronTrain using the dataset from task 3.

[Graph](averaged_perceptron.png)

Averaged perceptron converges at very fast and does not seem to overfit, at least until epoch 100. It also shows quite stable results (with(out) shuffling).

