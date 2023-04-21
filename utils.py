import numpy as np
from cluster import Cluster

def mahalanobis(x, cluster: Cluster) -> int:
    """
    Calculates the Mahalanobis distance between two vectors x and y, with covariance matrix cov.
    """
    x_minus_ref = np.array(x) - np.array(cluster.mean)
    inv_cov = np.linalg.inv(cluster.cov)
    return np.dot(np.dot(x_minus_ref.T, inv_cov), x_minus_ref)

def mahalanobis_cluster_to_vectors(vector_set, cluster: Cluster) -> int:
    """
    Calculates the sum of the Mahalanobis distances between the vectors and the reference vector.
    """
    distance = 0
    for vector in vector_set:
        distance += mahalanobis(vector, cluster)
    return distance

def mahalanobis_clusters_to_vectors(vector_set, clusters: list[Cluster]) -> int:
    distance = 0
    for cluster in clusters:
        distance += mahalanobis_cluster_to_vectors(vector_set, cluster)
    return distance

def find_cluster(vector_set) -> Cluster:
    """
    Calculates the cluster centre for a given vector set.
    """
    vector_set = np.array(vector_set)
    
    mean = vector_set.sum(axis=0)/vector_set.shape[0]
    
    cov = np.zeros((vector_set.shape[1], vector_set.shape[1]))
    for vector in vector_set:
        cov += np.outer((vector - mean), (vector - mean))
    cov = cov/vector_set.shape[0]
    
    return Cluster(mean, cov)

def assign_vectors_to_clusters(vector_set, clusters: list[Cluster]):
    """
    Assigns each vector in the vector set to one of the clusters. 
    Clusters is a list of Cluster objects.
    """
    for vector in vector_set:
        distances = [mahalanobis(vector, cluster) for cluster in clusters]
        assigned_cluster_idx = distances.index(min(distances))
        
        assigned_cluster = clusters[assigned_cluster_idx]
        assigned_cluster.assign_vectors(vector)
                
def split_clusters(clusters: list[Cluster]) -> list[Cluster]:
    """
    Finds the cluster with most assigned vectors, and creates a new cluster based on this cluster.
    """
    vector_per_cluster = []
    for cluster in clusters:
        vector_per_cluster.append(cluster.get_number_of_assigned_vectors())
        
    cluster_to_split_idx = vector_per_cluster.index(min(vector_per_cluster))
    cluster_to_split = clusters[cluster_to_split_idx]
    
    new_cluster_mean = cluster_to_split.mean*1.01
    new_cluster_cov = cluster_to_split.cov
    new_cluster = Cluster(new_cluster_mean, new_cluster_cov)
    
    return cluster + [new_cluster]

def find_clusters(vector_set):
    """TODO: Implement outer loop for different number of cluster. 
       TODO: For loop for 5 cluster, or while loop?"""
    number_of_clusters = 1
    clusters = [find_cluster(vector_set)]
    assign_vectors_to_clusters(vector_set, clusters)

    accumulated_distance = mahalanobis_clusters_to_vectors(vector_set, clusters)
    
    number_of_clusters += 1
    clusters = split_clusters(clusters)
    
    distance = np.inf
    while True:
        assign_vectors_to_clusters(vector_set, clusters)
        new_distance = mahalanobis_clusters_to_vectors(vector_set, clusters)
        
        if new_distance < distance:
            for idx, cluster in enumerate(clusters):
                cluster.update_cluster()
            distance = new_distance
        else:
            new_accumulated_distance = distance
            break
    
    if new_accumulated_distance < accumulated_distance:
        accumulated_distance = new_accumulated_distance