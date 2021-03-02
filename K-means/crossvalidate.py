from utils import *
from problem_generator import medium_y, hard_y, y_true, FIG_SIZE
import numpy as np
import matplotlib.pyplot as plt

centroids_true = find_centroids(hard_y, y_true)
samples = train_test_split_ids(hard_y, 10)

scores = []

for sample in samples:
    # train_test_split
    test = hard_y[sample]
    mask = np.ones(len(hard_y), dtype=bool)
    mask[sample] = False
    train = hard_y[mask]

    # train centroids
    _, centroids = kmeans_cluster_assignment(3, train)

    # apply centroids to test data
    clustered, centroids = kmeans_cluster_assignment(3, test, centroids)

    # map predicted and correct clusters
    map = map_clusters(centroids, centroids_true)

    preds = {cluster: sample[ids] for cluster, ids in clustered.items()}

    true = {cluster: [id for id in ids if id in sample] \
                        for cluster, ids in y_true.items()}

    scores.append(performance(true,preds,map))


fig = plt.figure(figsize=FIG_SIZE)
# plt.plot([i for i in range(1, 11)], scores, color='g',linewidth=2)
avg_performance = np.mean(scores)
plt.fill_between([i for i in range(0, 10)], scores, avg_performance, alpha=0.5)
# plt.legend(('Unsupervised learning', 'With trained centroids'))
plt.xlabel('Sample N')
plt.ylabel('Performance')
plt.grid(True)
plt.title('Performance of kmeans for different train_test_splits')
plt.savefig('plots/batches.pdf')
