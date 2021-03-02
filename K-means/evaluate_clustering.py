from utils import *
from problem_generator import medium_y, hard_y, y_true, FIG_SIZE
import numpy as np
import matplotlib.pyplot as plt

# peformance over iterations
data = []
for n in range(10,101,10):
    data.append(kmeans_cluster_assignment(3, hard_y, max_iterations=n))

centroids_true = find_centroids(hard_y, y_true)

scores = []
for preds, centroids_preds in data:
    scores.append(performance(
        y_true,
        preds,
        map_clusters(centroids_preds, centroids_true)))

# plot both sub tasks at the end

# trained centroids over iterations
proportion = [int(hard_y.shape[0]*0.9)]
train_ids, test_ids = train_test_split_ids(hard_y, proportion)

data = []
for n in range(10,101,10):
    data.append(kmeans_cluster_assignment(3, hard_y[train_ids], max_iterations=n))

tests = []
for _, centroids in data:
    tests.append(kmeans_cluster_assignment(k=3,
                                           points=hard_y[test_ids],
                                           centers_guess = centroids))


trained_scores = []
for test in tests:
    map = map_clusters(test[1], centroids_true)

    # filter preds and y_true for test ids only
    preds = {cluster: test_ids[ids] for cluster, ids in test[0].items()}
    true = {cluster: [id for id in ids if id in test_ids] \
                    for cluster, ids in y_true.items()}

    trained_scores.append(performance(true,preds,map))

fig = plt.figure(figsize=FIG_SIZE)
plt.plot([i for i in range(10, 101, 10)], scores, color='g',linewidth=2)
plt.plot([i for i in range(10, 101, 10)], trained_scores, color='b',linewidth=2)
plt.legend(('Unsupervised learning', 'With trained centroids'))
plt.xlabel('Iterations')
plt.ylabel('Performance')
plt.grid(True)
plt.title('Performance of kmeans over a number of iterations')
plt.savefig('plots/iterations.pdf')
