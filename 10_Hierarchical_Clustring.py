from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X , Y = make_blobs(n_samples=100, centers=4, random_state=42)

linkage_matrix = linkage(X, method='ward')

plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

clustes = AgglomerativeClustering(n_clusters=4,  linkage='ward')
y_red =  clustes.fit(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_red, cmap='viridis', marker='o')

plt.title('Hierarchical Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster Label')
plt.show()