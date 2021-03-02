from scipy.cluster.vq import vq, kmeans, whiten
from problem_generator import hard_y
from utils import kmeans_cluster_assignment
import timeit

print('Wait for 1000 iterations to be over\n')

print('The average time to execute my kmeans is ',
    timeit.timeit('kmeans_cluster_assignment(3, hard_y)',
                  number=1000,
                  globals=globals()) / 1000)

print('The average time to execute scipy kmean is ',
    timeit.timeit('kmeans(whiten(hard_y), 3)',
                  number=1000,
                  globals=globals()) / 1000)
