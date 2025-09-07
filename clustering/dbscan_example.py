import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)
# make_moons(): A function from sklearn.datasets that creates moon-shaped clusters
# X: A 2D numpy array with shape (300, 2) containing the coordinates
# _: The ignored labels (0 and 1 indicating which moon each point belongs to)
# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)
print(labels)
# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.title("DBSCAN Clustering")
plt.show()

print("Number of clusters:", len(set(labels)) - (1 if -1 in labels else 0))
print("Number of noise points:", list(labels).count(-1))



from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

# Generate sample data
X1, y1 = make_moons(n_samples=300, noise=0.05, random_state=0)
X2, y2 = make_blobs(n_samples=100, centers=[[3, 3]], cluster_std=0.5, random_state=0)
X = np.vstack([X1, X2])
# ytrue = np.vstack((y1, y2))

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Get cluster information
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters)
print('Estimated number of noise points: %d' % n_noise)

# Visualize the results
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

plt.figure(figsize=(12, 5))

# Plot original data
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Original Data")

# Plot clustered data
plt.subplot(1, 2, 2)
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    # Plot core points
    xy = X_scaled[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10, alpha=0.8)

    # Plot non-core points
    xy = X_scaled[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6, alpha=0.6)

plt.title('DBSCAN Clustering\nEstimated number of clusters: %d' % n_clusters)
# \n - newline character to create a subtitle
# %d - placeholder for an integer value
# % n_clusters - String formatting that replaces %d with the value of n_clusters
plt.show()



# Evaluate clustering performance (if ground truth is available)
if len(set(y_true)) > 1:  # Only if we have true labels
    print("Adjusted Rand Index:", metrics.adjusted_rand_score(y_true, labels))
    print("Silhouette Coefficient:", metrics.silhouette_score(X, labels))

# For datasets without ground truth
print("Silhouette Score:", metrics.silhouette_score(X_scaled, labels))





from sklearn.neighbors import NearestNeighbors
import numpy as np

# Method to find optimal eps value
def find_optimal_eps(X, min_samples):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, min_samples-1], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-distance Graph')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel('Epsilon')
    plt.show()
    
    # The "elbow" point is a good candidate for eps
    return distances[int(len(distances)*0.95)]  # 95th percentile as suggestion

# Find optimal parameters
min_samples = 5
optimal_eps = find_optimal_eps(X_scaled, min_samples)
print(f"Suggested eps value: {optimal_eps:.3f}")

# Run DBSCAN with suggested parameters
dbscan_optimal = DBSCAN(eps=optimal_eps, min_samples=min_samples)
labels_optimal = dbscan_optimal.fit_predict(X_scaled)




from sklearn.cluster import KMeans

# Compare with K-Means
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', s=50)
plt.title("True Labels")

plt.subplot(1, 3, 2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50)
plt.title("DBSCAN Clustering")

plt.subplot(1, 3, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=50)
plt.title("K-Means Clustering")

plt.tight_layout()
plt.show()