from sklearn.cluster import KMeans
import numpy as np

# Number of clusters
kmeans = KMeans(n_clusters=3)

X = np.array([1,2,3,4,5,6,7,8,9])
X = X.reshape(-1, 1)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

print(centroids)
print(labels)
