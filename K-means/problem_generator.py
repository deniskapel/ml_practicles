from utils import scatter_clusters, plot_clusters
import matplotlib.pyplot as plt
N_POINTS = 100
FIG_SIZE = (10, 5)

# Task 1 Generate
easy_y = scatter_clusters([[0.5, 0.7], [1.5, 0.7], [1, 1.7]],
                        [0.2, 0.2],
                        N_POINTS)

medium_y = scatter_clusters([[0.5, 0.7], [1.5, 0.7], [1, 1.7]],
                        [.55, 0.55],
                        N_POINTS)

hard_y = scatter_clusters([[0.5, 0.7], [1.5, 0.7], [1, 1.7]],
                          [.75, 0.75],
                          N_POINTS)

y_true ={i: [idx for idx in range(i*N_POINTS,N_POINTS+i*N_POINTS)] for i in range(3)}


# Task 1 Plot
fig = plt.figure(figsize=FIG_SIZE)
easy_plot = plot_clusters(fig, easy_y, y_true)
easy_plot.savefig("plots/easy_true.pdf", bbox_inches='tight')

fig = plt.figure(figsize=FIG_SIZE)
medium_plot = plot_clusters(fig, medium_y, y_true)
medium_plot.savefig("plots/medium_true.pdf", bbox_inches='tight')

fig = plt.figure(figsize=FIG_SIZE)
hard_plot = plot_clusters(fig, hard_y, y_true)
hard_plot.savefig("plots/hard_true.pdf", bbox_inches='tight')

# plt.sow() # plots only the last problem, move up to see others
