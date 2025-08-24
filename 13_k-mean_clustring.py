import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X,y = make_blobs(n_samples=100, centers=3, random_state=42)

kmeans_model  = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X)

cluster_center = kmeans_model.cluster_centers_

label = kmeans_model.labels_

plt.scatter(X[:, 0], X[:, 1], c=label, cmap='viridis', edgecolors='k', s=50, marker='o')
plt.scatter(cluster_center[:, 0], cluster_center[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
plt.legend()
plt.show()