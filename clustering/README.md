# K-Means Clustering Implementation

This repository contains Python code demonstrating K-Means clustering algorithm implementation using both synthetic and real-world datasets. The code showcases data generation, clustering, visualization, and evaluation techniques.

## Features

- **Synthetic Data Clustering**: Generate and cluster sample data using `make_blobs`
- **Real Dataset Analysis**: Apply K-Means to the famous Iris dataset
- **Visualization**: Plot clustering results and centroids
- **Elbow Method**: Determine optimal number of clusters
- **Performance Evaluation**: Use confusion matrix and classification reports

## Requirements

```python
pip install numpy matplotlib scikit-learn
```

## Code Structure
**Part 1: Synthetic Data Clustering**
1. **Data Generation** using make_blobs
Visualizes the unlabeled data with true cluster colors

2. **K-Means Implementation**
- Fits the model to the data
- Extracts and displays:
    - Cluster centers
    - Point labels
    - Inertia (sum of squared distances)
    - Number of iterations

3. **Results Visualization**
-Plots clustered data points with different colors
-Marks cluster centroids with red X markers

4. **Elbow Method Analysis**
- Tests K values from 1 to 10
- Plots inertia vs. number of clusters
- Helps determine optimal cluster count

**Part 2: Iris Dataset Analysis**
1. **Data Preparation**
Applies StandardScaler for feature normalization

2. **Clustering Application**
- Applies K-Means with 3 clusters (matching Iris species count)
- Predicts cluster labels

3. **Performance Evaluation**
- Generates **confusion matrix comparing true vs. predicted labels**
- **Provides classification report with precision, recall, and F1-score**

4. **2D Visualization**
- Plots clusters using first two features (sepal length and width)
- Displays cluster centroids

## Key Outputs
1. **Cluster Centers**: Coordinates of each cluster's centroid
2. **Labels**: Cluster assignment for each data point
3. **Inertia**: Measure of clustering quality (lower is better)
4. Iterations: Number of algorithm iterations required
5. **Evaluation** Metrics: Accuracy, precision, recall, F1-score

## Usage
1. Run the script to see synthetic data clustering
2. Observe the elbow method plot to understand optimal cluster selection
3. Review Iris dataset clustering results and performance metrics
4. Modify parameters (n_clusters, random_state) to experiment with different configurations

**Key Concepts Demonstrated**

1. **Unsupervised Learning**: Clustering without predefined labels
2. **Feature Scaling**: Importance of normalizing features
3. **Cluster Validation**: Using inertia and evaluation metrics
4. **Dimensionality Reduction**: 2D visualization of multi-dimensional data

**Parameters Explained**
n_clusters: Number of clusters to form

random_state: Seed for reproducible results

n_init: Number of times algorithm runs with different centroid seeds

cluster_std: Standard deviation of clusters in synthetic data

## Visualizations
1. Scatter plots with color-coded clusters
2. Centroid markers (red X)
3. Elbow method curve for optimal K selection
4. 2D projections of multi-dimensional data

**Notes**
The elbow method helps determine optimal cluster count when unknown

Feature scaling is crucial for real-world datasets with varying scales

K-Means assumes spherical clusters and may not perform well with complex shapes

Results may vary with different random seeds