from __future__ import annotations
import numpy as np
from cluster import Cluster
    
def mahalanobis(x, cluster: Cluster) -> int:
    """
    Calculates the Mahalanobis distance between two vectors x and y, with covariance matrix cov.
    """
    x_minus_ref = np.array(x) - np.array(cluster.mean)
    
    u, s, v = np.linalg.svd(cluster.cov)
    inv_cov = np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose()))
    
    return np.dot(np.dot(x_minus_ref.T, inv_cov), x_minus_ref)

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

def divide_vector_set(vector_set, n):
    """ 
    Divide vector set into chunks of size n.
    """
    for i in range(0, len(vector_set), n):
        yield vector_set[i:i + n]
