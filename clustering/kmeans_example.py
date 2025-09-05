import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
# Part 1. Sample data
# 1. Generate sample data (default number of features of each sample is 2)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
# Visualize the unlabeled data
plt.scatter(X[:,0], X[:,1], c=y_true, s = 50, edgecolors='grey', alpha=0.8) # s is the marker size
plt.title("Generated Data for Clustering")
plt.show()

# 2. Apply K-Means 
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
# trial 1: Fit the model
kmeans.fit(X)

# Access the learned attributes
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Labels for each point:", kmeans.labels_)
print("Sum of squared distances to closest centroid (Inertia):", kmeans.inertia_)
print("Number of iterations run:", kmeans.n_iter_)
# *   `cluster_centers_` (ndarray of shape (n_clusters, n_features)): Coordinates of cluster centers.
# *   `labels_` (ndarray of shape (n_samples,)): Label (cluster index) of each point.
# *   `inertia_` (float): Sum of squared distances of samples to their closest cluster center. This is the value the algorithm tries to minimize. A lower inertia means a better fit (but beware of overfitting with high K!).
# *   `n_iter_` (int): Number of iterations run for that particular initialization.

kmeans.transform(X)
#    `.transform(X)`: Transforms `X` to a cluster-distance space. Returns an array 
# where each element is the distance from the sample to each cluster center `(n_samples, n_clusters)`.


# trial 2: fit the model to the data and predict the cluster labels.
labels = kmeans.fit_predict(X)
# Get the coordinates of the cluster centers
centroids = kmeans.cluster_centers_

# 3. Visualize the results
plt.scatter(X[:,0], X[:,1], c = labels, s=50, cmap='viridis')
# PLOT centroids
plt.scatter(centroids[:,0], centroids[:,1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering Results")
plt.show()
print(labels)
print("Cluster centroids:\n", centroids)


# 4. The hardest part of K-Means is choosing the right number of clusters if you don't know it beforehand. The Elbow Method is a common technique to help with this.

# Calculate inertia for different values of k
inertia = []
k_range = range(1, 11) # Testing k from 1 to 10

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_) # Inertia: Sum of squared distances to closest centroid

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.xticks(k_range)
plt.grid(True)
plt.show()



# Part 2. Real Dataset: Iris dataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# 2. Apply K-Means
kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_iris = kmeans_iris.fit_predict(X_iris_scaled)

# 3. Compare the predicted clusters with the actual species
# use a contingency matrix to see the relationships.
from sklearn.metrics import confusion_matrix, classification_report
print("Contingency Matrix:")
print(confusion_matrix(y_iris, labels_iris))
print("\nClassification Report:")
print(classification_report(y_iris, labels_iris))

# 4. Visualiza the clusters using the first two features for 2D plot
plt.scatter(X_iris_scaled[:,0], X_iris_scaled[:,1], c=labels_iris, s= 50, cmap='virdis')
plt.scatter(kmeans_iris.cluster_centers_[:,0], kmeans_iris.cluster_centers_[:,1], c='red', s=200, alpha=0.75, marker="X")
plt.xlabel(iris.feature_name[0])
plt.ylabel(iris.feature_name[1])
plt.show()