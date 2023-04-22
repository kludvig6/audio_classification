import numpy as np

class Cluster:
    def __init__(self, _mean, _cov):
        "TODO: Check size error on mean and cov?"
        self.mean = _mean
        self.cov = _cov
        
        self.assigned_vectors = []
     
    def assign_vectors(self, vector_set):
        """
        Assigns a vector or a set of vectors to this cluster by extending og appending to the class variable assigned_vectors.
        """
        "TODO: Compare size of vectors and mean?"
        if isinstance(vector_set[0], list):
            self.assigned_vectors.extend(vector_set)
        else:
            self.assigned_vectors.append(vector_set)
    
    def get_number_of_assigned_vectors(self):
        """
        Returns the number of assigned vectors in the cluster.
        """
        return len(self.assigned_vectors)

    def update_cluster(self):
        """
        This is called after new vectors have been assigned. Given the new assigned vectors, it updates the mean and covariance of the cluster.
        """
        vector_set = np.array(self.assigned_vectors)
    
        self.mean = vector_set.sum(axis=0)/vector_set.shape[0]
    
        cov = np.zeros((vector_set.shape[1], vector_set.shape[1]))
        for vector in vector_set:
            cov += np.outer((vector - self.mean), (vector - self.mean))
        self.cov = cov/vector_set.shape[0]
        
