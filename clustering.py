from __future__ import annotations
import numpy as np
import copy

from cluster import Cluster
from utils import mahalanobis, mahalanobis_all_clusters, divide_vector_set

def assign_vectors_to_clusters(vector_set, clusters: list[Cluster]):
    """
    Clears all assigned vectors, and assigns each vector in the vector set to one of the clusters. 
    """
    for cluster in clusters:
        cluster.assigned_vectors = []
        
    for vector in vector_set:
        vector_copy = copy.copy(vector)
        
        distances = [mahalanobis(vector_copy, cluster) for cluster in clusters]  
        assigned_cluster_idx = distances.index(min(distances))
        
        assigned_cluster = clusters[assigned_cluster_idx]
        assigned_cluster.assign_vectors(vector_copy)
            
                
def create_new_cluster(clusters: list[Cluster]) -> list[Cluster]:
    """
    Finds the cluster with most assigned vectors, and creates a new cluster just next to it. Return the list of clusters with the new cluster included.
    """    
    vector_per_cluster = [
        cluster.get_number_of_assigned_vectors() for cluster in clusters
    ]
    
    cluster_to_split_idx = vector_per_cluster.index(max(vector_per_cluster))
    cluster_to_split = clusters[cluster_to_split_idx]

    new_cluster_mean = copy.copy(cluster_to_split.mean)
    new_cluster_cov = copy.copy(cluster_to_split.cov)
    new_cluster_genre = copy.copy(cluster_to_split.genre)
    
    new_cluster = Cluster(new_cluster_mean, new_cluster_cov, new_cluster_genre)
    new_cluster.randomly_move_centre()
    
    return clusters + [new_cluster]

def cluster_from_vectors(vector_set, genre) -> Cluster:
    """
    Calculates the cluster centre for a given vector set.
    """
    vector_array = np.array(vector_set, copy=True)
    
    mean = vector_array.sum(axis=0)/vector_array.shape[0]
    
    cov = np.zeros((vector_array.shape[1], vector_array.shape[1]))
    for vector in vector_array:
        cov += np.outer((vector - mean), (vector - mean))
    cov = cov/vector_array.shape[0]
    
    return Cluster(mean, cov, genre)

def some_cluster_empty(clusters: list[Cluster]):
    for cluster in clusters:
        if cluster.is_empty():
            return True
    return False

def find_five_clusters(vector_set, genre):
    """
    This functions returns the optimal five clusters for the given vector set.
    """
    MIN_REDUCTION = 0.99

    clusters = [cluster_from_vectors(vector_set, genre)]
    assign_vectors_to_clusters(vector_set, clusters)

    for i in range(5):
        clusters = create_new_cluster(clusters)
        distance_m_clusters = np.inf
        while True:
            assign_vectors_to_clusters(vector_set, clusters)
            while some_cluster_empty(clusters):
                for cluster in clusters:
                    print(cluster.get_number_of_assigned_vectors())   
                    if cluster.is_empty():
                        cluster.randomly_move_centre()
                print()
                assign_vectors_to_clusters(vector_set, clusters)        
                
            new_distance_m_clusters = mahalanobis_all_clusters(vector_set, clusters)

            if new_distance_m_clusters < MIN_REDUCTION*distance_m_clusters:
                for cluster in clusters:
                    cluster.update_cluster()
                    if cluster.get_number_of_assigned_vectors() == 1:
                        cluster.randomly_move_centre()
                distance_m_clusters = new_distance_m_clusters
            else:
                break
            
    return clusters
        
def new_find_five_clusters(vector_set, genre):
    """
    This functions returns the optimal five clusters for the given vector set.
    """
    MIN_REDUCTION = 0.99
    NUMBER_OF_CLUSTERS = 5

    divided_vector_set = np.array_split(vector_set,NUMBER_OF_CLUSTERS)

    clusters = []
    for vectors in divided_vector_set:
        cluster = cluster_from_vectors(vectors, genre)
        cluster.assign_vectors(vectors)
        clusters.append(cluster)

    
    distance_m_clusters = np.inf
    print("Number of clusters:", len(clusters))
    while True:
        assign_vectors_to_clusters(vector_set, clusters)
        while some_cluster_empty(clusters):
            for cluster in clusters:
                print(cluster.get_number_of_assigned_vectors())   
                if cluster.is_empty():
                    cluster.randomly_move_centre()
            print()
            assign_vectors_to_clusters(vector_set, clusters)   
                
        new_distance_m_clusters = mahalanobis_all_clusters(vector_set, clusters)

        if new_distance_m_clusters < MIN_REDUCTION*distance_m_clusters:
            for cluster in clusters:
                cluster.update_cluster()
            distance_m_clusters = new_distance_m_clusters
        else:
            break
            
    return clusters


'''def find_all_clusters(vector_set, genre):
    """
    This functions calculates on it's own the optimal number of clusters, and returns these clusters.
    """
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
            
            while empty_cluster(clusters):
                print("empty cluster")
                for cluster in clusters:
                    if cluster.get_number_of_assigned_vectors() == 0:
                        cluster.randomly_move_centre()
                assign_vectors_to_clusters(vector_set, clusters)
                
            new_distance_m_clusters = mahalanobis_all_clusters(vector_set, clusters)

            if new_distance_m_clusters < MIN_REDUCTION*distance_m_clusters:
                for idx, cluster in enumerate(clusters):
                    print(cluster.assigned_vectors)
                    cluster.update_cluster()
                distance_m_clusters = new_distance_m_clusters
            else:
                new_lowest_distance = distance_m_clusters
                break
            
        if new_lowest_distance < MIN_REDUCTION*currently_lowest_distance:
            currently_lowest_distance = new_lowest_distance
        else:
            return clusters'''