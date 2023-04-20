import numpy as np

def mahalanobis(x, mean, cov):
    """
    Calculates the Mahalanobis distance between two vectors x and y, with covariance matrix cov.
    """
    x_minus_ref = np.array(x) - np.array(mean)
    inv_cov = np.linalg.inv(cov)
    return np.dot(np.dot(x_minus_ref.T, inv_cov), x_minus_ref)

def accumulated_mahalonobis(vector_set, mean, cov):
    """
    Calculates the sum of the Mahalanobis distances between the vectors and the reference vector.
    """
    distance = 0
    for vector in vector_set:
        distance += mahalanobis(vector, mean, cov)
    return distance

def calculate_reference_vector(vector_set):
    """
    Calculates the reference vector for a cluster of vectors.
    """
    vector_set = np.array(vector_set)
    
    mean = vector_set.sum(axis=0)/vector_set.shape[0]
    
    cov = np.zeros((vector_set.shape[1], vector_set.shape[1]))
    for vector in vector_set:
        cov += np.outer((vector - mean), (vector - mean))
    cov = cov/vector_set.shape[0]
    
    return mean, cov


def find_clusters(vector_set):
    M = 1
    mean, cov = calculate_reference_vector(vector_set)
    accumulated_distance = accumulated_mahalonobis(vector_set, mean, cov)
    M += 1

def assign_vectors_to_clusters(vector_set, cluster_references):
    """
    Assigns each vector in the vector set to one of the clusters. 
    Clusters is given by [mean, cov].
    """
    for vector in vector_set:
        assigned_cluster = cluster_references[0]
        assigned_mean, assigned_cov = assigned_cluster
        assigned_distance = mahalanobis(vector, assigned_mean, assigned_cov)
        for cluster in cluster_references[1:]:
            mean, cov = cluster
            distance = mahalanobis(vector, mean, cov)   
            if mahalanobis(vector, mean, cov) < assigned_distance:
                
def split_clusters(vector_set, cluster_references):
    