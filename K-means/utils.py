import matplotlib.colors as mcolors
import sys
import numpy as np
from typing import List, TypedDict, Optional

EMPTY = "\nOne of the clusters is empty. \
Please restart to reassign centers_guess \
or change the number of clusters.\n\n"
COLORS = list(mcolors.BASE_COLORS.keys())

Vector = List[float]
Vectors = List[Vector]
class Clusters(TypedDict):
    int: Vectors

def scatter_clusters(centers: Vectors,
                     spread: Vector,
                     n_points: int) -> Vectors:
    """
        It generates [[X1,Y1], [Xn, Yn]] for given clusters centers

        output[0:n_points] -> 1st cluster
        output[n-1_points:n_points] -> nth cluster
    """
    x = []
    y = []
    for center in centers:
        x.extend([np.random.uniform(center[0]-spread[0],center[0]+spread[0]) for i in range(n_points)])
        y.extend([np.random.uniform(center[1]-spread[1],center[1]+spread[1]) for i in range(n_points)])

    # return np.asarray([point for point in zip(x, y)]).reshape(len(centers),-1,2)
    return np.column_stack((x,y)) # [[x1,y1], [x2,y2]]


def plot_clusters(fig, points: Vectors, clusters: Clusters):
    """
        creates and returns a plt object based on provided points.
        They are coloured based on clusters dict
    """
    ax=fig.add_axes([0,0,1,1])

    for i, ids in enumerate(clusters.values()):
        ax.scatter(points[ids, 0], points[ids, 1], color=COLORS[i])

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Clusters')
    ax.grid(True)

    return fig


def kmeans_cluster_assignment(
    k: int,
    points: Vectors,
    centers_guess: Optional[Vectors] = None,
    max_iterations: Optional[int] = 100,
    tolerance: Optional[float] = 0.001
    ) -> Clusters:
    """ clustering with a kmeans algorith from scratch """
    if centers_guess is None:
        centers_guess = random_centroids(k, points)

    for i in range(max_iterations):
        clusters = {cluster: [] for cluster in range(k)}

        for id, point in enumerate(points):
            # calculate distance to each centroid
            cluster = np.argsort(
                [calc_dist(point, center) for center in centers_guess]
                )[0]

            clusters[cluster].append(id)

        # no_empty_or_break(clusters)

        centroids = find_centroids(points, clusters)

        if calc_dist(centers_guess, centroids) < tolerance:
            # break if centroids stoppped moving
            break
        centers_guess = centroids

    return clusters, centers_guess


def calc_dist(point1: Vector, point2: Vector) -> float:
    """ calc distance between 2 points """
    return np.linalg.norm(point2-point1)

def random_centroids(k: int, points: Vectors) -> Vectors:
    """ generate k random 2D points using given points for a range """
    return np.random.normal(np.mean(points, axis=0),
                            np.std(points, axis=0),
                            [k,points.shape[1]])


def find_centroids(points: Vectors, clusters: Clusters) -> Vectors:
    """ pass points and calculate centroids for given clusters """
    centroids = np.array(
        [np.mean(points[ids], axis=0) for ids in clusters.values()])

    return centroids


def no_empty_or_break(clusters: dict) -> None:
    """ stops the algorithm if one of the clusters is empty """
    for value in clusters.values():
        if len(value) == 0:
            sys.stderr.write(EMPTY)
            sys.exit(1)


def sort_clusters(clusters: Clusters) -> Clusters:
    """ sort clusters, so the first cluster is has the smallest mean"""
    return sorted(clusters, key=cluster_mean)


# re-order clusters
def map_clusters(preds: Clusters, true: Clusters):
    """ create a mapping for clusters based on their centroids """
    map = {}
    for i, centroid in enumerate(preds):
        map[i] = np.argsort([calc_dist(centroid, center) for center in true])[0]
    return map


def performance(y_true: Clusters, preds: Clusters, mapping: dict) -> float:
    """
        calculate average performance score for all clusters
        performance = average percent of points assigned the correct cluster
    """
    scores = []
    for cluster, ids in preds.items():
        true = y_true[mapping[cluster]]
        max = len(true)
        scores.append( # append percentage of intersecting ids
            np.intersect1d(ids, true).shape[0] / max)

    return np.mean(scores, dtype='float16') # return everage


def train_test_split_ids(dataset: Vectors, indices_or_sections=2) -> Vectors:
    """
        splits a datasets indexes into n groups shuffling it first,
        return a n lists of int

        indices_or_sectionsint or 1-D array
        If indices_or_sections is an integer, N, the array will be divided
        into N equal arrays along axis.
        If such a split is not possible, an error is raised.

        If indices_or_sections is a 1-D array of sorted integers,
        the entries indicate where along axis the array is split.
        For example, [2, 3] would, for axis=0, result in
            ary[:2]
            ary[2:3]
            ary[3:]

        If an index exceeds the dimension of the array along axis,
        an empty sub-array is returned correspondingly.

        for more on indices_sections see numpy documentation
        https://numpy.org/doc/stable/reference/generated/numpy.split.html
    """
    # get indexes
    ids = np.arange(dataset.shape[0])
    np.random.shuffle(ids)
    samples = []

    for sample in np.split(ids, indices_or_sections):
        yield sample
