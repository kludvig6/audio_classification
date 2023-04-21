import numpy as np

from utils import find_cluster

class Cluster:
    def __init__(self, _mean, _cov):
        "TODO: Check size error on mean and cov?"
        self.mean = _mean
        self.cov = _cov
        
        self.assigned_vectors = []
     
    def assign_vectors(self, vector_set):
        """
        Assigns a vector or a set of vectors to the cluster.
        """
        "TODO: Compare size of vectors and mean?"
        if isinstance(vector_set[0], list):
            self.assigned_vectors.extend(vector_set)
        else:
            self.assigned_vectors.append(vector_set)
    
    def get_number_of_assigned_vectors(self):
        return len(self.assigned_vectors)

    def update_cluster(self):
        vector_set = np.array(self.assigned_vectors)
    
        self.mean = vector_set.sum(axis=0)/vector_set.shape[0]
    
        cov = np.zeros((vector_set.shape[1], vector_set.shape[1]))
        for vector in vector_set:
            cov += np.outer((vector - self.mean), (vector - self.mean))
        self.cov = cov/vector_set.shape[0]