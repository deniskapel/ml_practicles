from typing import List, Tuple, Optional
import numpy as np
from utils import *
from problem_generator import y_true, easy_y, medium_y, hard_y, N_POINTS, FIG_SIZE
import matplotlib.pyplot as plt


# Task 2 Cluster and plot generated problems
# Test on easy
fig = plt.figure(figsize=FIG_SIZE)
easy_preds, _ = kmeans_cluster_assignment(3, easy_y)
easy_plt = plot_clusters(fig, easy_y, easy_preds)
easy_plt.savefig("plots/easy_25.pdf", bbox_inches='tight')

# medium 25 iterations
fig = plt.figure(figsize=FIG_SIZE)
medium_preds, _ = kmeans_cluster_assignment(3, medium_y, max_iterations=25)
medium_plt = plot_clusters(fig, medium_y, medium_preds)
medium_plt.savefig("plots/medium_25.pdf", bbox_inches='tight')

# medium 50 iterations
fig = plt.figure(figsize=FIG_SIZE)
medium_preds, _ = kmeans_cluster_assignment(3, medium_y, max_iterations=50)
medium_plt = plot_clusters(fig, medium_y, medium_preds)
medium_plt.savefig("plots/medium_50.pdf", bbox_inches='tight')

# medium 75 iterations
fig = plt.figure(figsize=FIG_SIZE)
medium_preds, _ = kmeans_cluster_assignment(3, medium_y, max_iterations=75)
medium_plt = plot_clusters(fig, medium_y, medium_preds)
medium_plt.savefig("plots/medium_75.pdf", bbox_inches='tight')

#  medium 100 iterations
fig = plt.figure(figsize=FIG_SIZE)
medium_preds, _ = kmeans_cluster_assignment(3, medium_y)
medium_plt = plot_clusters(fig, medium_y, medium_preds)
medium_plt.savefig("plots/medium_100.pdf", bbox_inches='tight')

# hard 25 iterations
fig = plt.figure(figsize=FIG_SIZE)
hard_preds, _ = kmeans_cluster_assignment(3, hard_y, max_iterations=25)
hard_plt = plot_clusters(fig, hard_y, hard_preds)
hard_plt.savefig("plots/hard_25.pdf", bbox_inches='tight')

# hard 50 iterations
fig = plt.figure(figsize=FIG_SIZE)
hard_preds, _ = kmeans_cluster_assignment(3, hard_y, max_iterations=50)
hard_plt = plot_clusters(fig, hard_y, hard_preds)
hard_plt.savefig("plots/hard_50.pdf", bbox_inches='tight')

# hard 75 iterations
fig = plt.figure(figsize=FIG_SIZE)
hard_preds, _ = kmeans_cluster_assignment(3, hard_y, max_iterations=75)
hard_plt = plot_clusters(fig, hard_y, hard_preds)
hard_plt.savefig("plots/hard_75.pdf", bbox_inches='tight')

# hard 100 iterations
fig = plt.figure(figsize=FIG_SIZE)
hard_preds, _ = kmeans_cluster_assignment(3, hard_y, max_iterations=100)
hard_plt = plot_clusters(fig, hard_y, hard_preds)
hard_plt.savefig("plots/hard_100.pdf", bbox_inches='tight')
