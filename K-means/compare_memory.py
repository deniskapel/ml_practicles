from problem_generator import hard_y, y_true
from scipy.cluster.vq import vq, kmeans, whiten
from utils import kmeans_cluster_assignment
import tracemalloc
import linecache
import os

def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

print('Trace scipy\n')
tracemalloc.start()
kmeans(whiten(hard_y), 3)
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)
tracemalloc.clear_traces()

print('\nTrace my kmeans\n')
tracemalloc.start()
kmeans_cluster_assignment(3, hard_y)
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)
tracemalloc.clear_traces()
