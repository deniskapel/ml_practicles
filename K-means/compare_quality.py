from problem_generator import hard_y, y_true, FIG_SIZE
from scipy.cluster.vq import vq, kmeans, whiten
from utils import kmeans_cluster_assignment, performance, map_clusters, find_centroids, calc_dist
from numpy import mean
import matplotlib.pyplot as plt

# cluster points
my_data = []
sy_data = []
for i in range(10):
    my_data.append(kmeans_cluster_assignment(3, whiten(hard_y)))
    sy_data.append(kmeans(whiten(hard_y), 3))


centroids_true = find_centroids(whiten(hard_y), y_true)

diff = []
for i in range(10):
    for true in centroids_true:
        my = calc_dist(centroids_true,sorted(my_data[i][1], key=mean))
        sy = calc_dist(centroids_true,sorted(sy_data[i][0], key=mean))
        diff.append(sy-my)


print(mean(diff))

fig = plt.figure(figsize=FIG_SIZE)
plt.xlabel('X_axis')
plt.ylabel('Y_axis')
plt.grid(True)
plt.title('Centroids - all points were whitened for consistency')

whitened = whiten(hard_y)
# Find 2 clusters in the data
codebook, distortion = kmeans(whitened, 3)
# Plot whitened data and cluster centers in red
plt.scatter(centroids_true[:, 0], centroids_true[:, 1], c='b')
plt.scatter(my_data[0][1][:, 0], my_data[0][1][:, 1], c='g')
plt.scatter(codebook[:, 0], codebook[:, 1], c='r')
plt.legend(('True', 'MyKmeans', 'SciPy'))
fig.savefig('plots/my_vs_scipy_quality.pdf')
