import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 1. Generate sample data (default number of features of each sample is 2)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
# Visualize the unlabeled data
plt.scatter(X[:,0], X[:,1], s = 50) # s is the marker size
plt.title("Generated Data for Clustering")
plt.show()

# 2. Apply K-Means 
