import numpy as np
from cluster import Cluster

def mahalanobis(x, cluster: Cluster) -> int:
    """
    Calculates the Mahalanobis distance between two vectors x and y, with covariance matrix cov.
    """
    try:
        x_minus_ref = np.array(x) - np.array(cluster.mean)
        inv_cov = np.linalg.inv(cluster.cov)
        return np.dot(np.dot(x_minus_ref.T, inv_cov), x_minus_ref)
    except:
        print("Mahalanobis, cov:", cluster.cov)

def mahalanobis_single_cluster(vector_set, cluster: Cluster) -> int:
    """
    Calculates the sum of the Mahalanobis distances between vectors and a cluster reference.
    """
    distance = 0
    for vector in vector_set:
        distance += mahalanobis(vector, cluster)
    return distance

def mahalanobis_all_clusters(vector_set, clusters: list[Cluster]) -> int:
    """
    Calculates the sum of the Mahalanobis distances between the clusters and their assigned vectors.
    """
    distance = 0
    for cluster in clusters:
        distance += mahalanobis_single_cluster(cluster.assigned_vectors, cluster)
    return distance
