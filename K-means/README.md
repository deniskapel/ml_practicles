# K-means clustering practical

## Task 1 Write a clustering problem generator with signature. For k=3, generate easy and hard problems and plot them; the easy problem might look like figure 3.13 from Daumé.

```python3
python3 problem_generator.py
```

It creates [plots](plots/) for easy_true, medium_true and hard_true problems generated for further tasks.

## Task 2 Implement K-means clustering as shown in Daumé. Replot your problems at 5 stages (random initialisation, 25%, 50%, 75%, 100% of iterations), using colours to assign points to clusters. The easy problem plots might look like the coloured plots in figure 3.14 from Daumé.

```python3
python3 cluster_points.py
```

Plots are saved into [plots folder](plots/) with the number of iterations indicated in the plot title. Clusters might have different shapes (e.g. [medium_100_reshaped.pdf](plots/medium_100_reshaped.pdf)) but generally clusters are formed properly

### Task 3 Study the performance of these two implementations: memory, speed, quality; compare against scipy.cluster.vq.kmeans.

First, comment line 67 in utils.py. It will prevent program shutdown if one of the clusters is empty.

**Speed**:
```python3
python3 compare_speed.py
```

Scipy is much faster

```
    The average time to execute my kmeans is  0.1056227034
    The average time to execute scipy kmean is  0.006389516700000001
```

**Memory**:
```python3
python3 compare_memory.py
```

Scipy is more effective as well. It also uses a fixed amount of allocated memory while my kmeans can use 15-30 KiB

```
    Trace scipy

    Top 3 lines
    #1: C:\Users\denis\AppData\Local\Programs\Python\Python38\lib\site-packages\scipy\cluster\vq.py:454: 1.0 KiB
        book, dist = _kmeans(obs, guess, thresh=thresh)
    #2: C:\Users\denis\AppData\Local\Programs\Python\Python38\lib\site-packages\numpy\lib\function_base.py:484: 0.6 KiB
        if a.dtype.char in typecodes['AllFloat'] and not np.isfinite(a).all():
    #3: <__array_function__ internals>:5: 0.5 KiB
    12 other: 2.5 KiB
    Total allocated size: 4.6 KiB

    Trace my kmeans

    Top 3 lines
    #1: C:\Users\denis\Documents\hse\Machine Learning\ml_practicles\K-means\utils.py:102: 12.4 KiB
        [np.mean(points[ids], axis=0) for ids in clusters.values()])
    #2: C:\Users\denis\AppData\Local\Programs\Python\Python38\lib\site-packages\numpy\linalg\linalg.py:2363: 2.3 KiB
        @array_function_dispatch(_norm_dispatcher)
    #3: <__array_function__ internals>:5: 2.0 KiB
    19 other: 7.0 KiB
    Total allocated size: 23.7 KiB
```

**Quality** to be completed when the function is ready
```python3
python3 compare_quality.py
```

Performance is compared through the difference between true centroids and predicted as scipy only returns centroids and distortion

The distance from scipy is subtracted from my_kmeans and the mean is taken. Generally, the differenc is not large ~0.07. All the data was whitened first for consistency.

The graph with calculated centroids is over [here](plots/my_vs_scipy_quality.pdf)s.

## Task 4 Compute the performance of your algorithm as percent of points assigned the correct cluster. (Your algorithm may order the clusters differently!) Graph this as a function of iterations, at 10% intervals. Additionally, make a random 10-90 test-train split; now you train on 90% of the data, and evaluate on the other 10%. How does the performance graph change?**

```python3
python3 evaluate_clustering.py
```

The medium and easy problems are not very interesting as they performance is ~.90 and 1. respectively.

The graph for performance over iterations available in [plots folder](plots/iterations.pdf). Unsupervised clustering score is ~70% most of the time: the line is jagged. With pre-trained centroids, the line is smooth and the score is generally higher: ~80 (if no empty clusters appear). The clusters of the hard problem intersect significantly, so the results might vary.

** The data splits are unbalanced --- a split that keeps cluster proportions might perform better **

## Task 5 Instead of a pure 10-90 split, divide your data into 10 portions. Picking one of these portions as test and the rest as train, we have 10 different 10-90 test-train splits. Each split gives a different train-eval run, and thus a different performance number. Perform cross-validation on your training data: plot the mean of these performances against percent-of-iterations. Error bars for these means are computed using standard deviation. Use filled plotting to show this region on the graph with matplotlib.pyplot.fill_between.


```python3
python3 crossvalidate.py
```

The graph is avaible [here](plots/batches.pdf). The results on some splits get to 90% while the others decrease to 70. Since each run, the performance varies (sometimes significantly), it might be more informatice If the splits are fixed and more balanced in terms of cluster distribution.
