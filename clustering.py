import numpy as np

from cluster import Cluster
from utils import mahalanobis, mahalanobis_all_clusters

def assign_vectors_to_clusters(vector_set, clusters: list[Cluster]):
    """
    Clears all assigned vectors, and assigns each vector in the vector set to one of the clusters. 
    """
    for cluster in clusters:
        cluster.assigned_vectors = []
        
    for vector in vector_set:
        distances = [mahalanobis(vector, cluster) for cluster in clusters]
        assigned_cluster_idx = distances.index(min(distances))
        
        assigned_cluster = clusters[assigned_cluster_idx]
        assigned_cluster.assign_vectors(vector)
            
                
def create_new_cluster(clusters: list[Cluster]) -> list[Cluster]:
    """
    Finds the cluster with most assigned vectors, and creates a new cluster just next to it. Return the list of clusters with the new cluster included.
    """
    vector_per_cluster = []
    for cluster in clusters:
        vector_per_cluster.append(cluster.get_number_of_assigned_vectors())
        
    cluster_to_split_idx = vector_per_cluster.index(max(vector_per_cluster))
    cluster_to_split = clusters[cluster_to_split_idx]
    
    random_seed = np.random.randint(-10, 10, len(cluster_to_split.mean))
    random_mean_multiplier = (random_seed/100) + 1

    new_cluster_mean = cluster_to_split.mean*random_mean_multiplier
    new_cluster_cov = cluster_to_split.cov
    new_cluster_genre = cluster_to_split.genre
    new_cluster = Cluster(new_cluster_mean, new_cluster_cov, new_cluster_genre)
    
    return clusters + [new_cluster]

def find_single_cluster(vector_set, genre) -> Cluster:
    """
    Calculates the cluster centre for a given vector set.
    """
    vector_set = np.array(vector_set)
    
    mean = vector_set.sum(axis=0)/vector_set.shape[0]
    
    cov = np.zeros((vector_set.shape[1], vector_set.shape[1]))
    for vector in vector_set:
        cov += np.outer((vector - mean), (vector - mean))
    cov = cov/vector_set.shape[0]
    
    return Cluster(mean, cov, genre)

def find_all_clusters(vector_set, genre):
    "TODO: Task specifically asks for 5 cluster. Implement this."
    "TODO: Remove while True and change to something more meaningful."
    MIN_REDUCTION = 0.99
    number_of_clusters = 1
    clusters = [find_single_cluster(vector_set, genre)]
    assign_vectors_to_clusters(vector_set, clusters)

    currently_lowest_distance = mahalanobis_all_clusters(vector_set, clusters)
    
    while True:
        number_of_clusters += 1
        clusters = create_new_cluster(clusters)

        distance_m_clusters = np.inf
        while True:
            assign_vectors_to_clusters(vector_set, clusters)
            new_distance_m_clusters = mahalanobis_all_clusters(vector_set, clusters)

            if new_distance_m_clusters < MIN_REDUCTION*distance_m_clusters:
                for idx, cluster in enumerate(clusters):
                    cluster.update_cluster()
                distance_m_clusters = new_distance_m_clusters
            else:
                new_lowest_distance = distance_m_clusters
                break
            
        if new_lowest_distance < MIN_REDUCTION*currently_lowest_distance:
            currently_lowest_distance = new_lowest_distance
        else:
            return clusters
    
    
def find_five_clusters(vector_set, genre):
    MIN_REDUCTION = 0.99
    number_of_clusters = 1
    clusters = [find_single_cluster(vector_set, genre)]
    assign_vectors_to_clusters(vector_set, clusters)

    while number_of_clusters < 5:
        number_of_clusters += 1
        clusters = create_new_cluster(clusters)

        distance_m_clusters = np.inf
        while True:
            assign_vectors_to_clusters(vector_set, clusters)
            new_distance_m_clusters = mahalanobis_all_clusters(vector_set, clusters)

            if new_distance_m_clusters < MIN_REDUCTION*distance_m_clusters:
                for idx, cluster in enumerate(clusters):
                    cluster.update_cluster()
                distance_m_clusters = new_distance_m_clusters
            else:
                break
            
    return clusters