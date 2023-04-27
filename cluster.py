import numpy as np
import copy

class Cluster:
    def __init__(self, _mean, _cov, _genre):
        self.mean = _mean
        self.cov = _cov
        
        self.assigned_vectors = []
        self.genre = _genre
    
    def __str__(self):
        return "Genre: " + str(self.genre) + "\nMean: " + str(self.mean) + "\nCov: " + str(self.cov)
    
    def assign_vectors(self, vector_set):
        """
        Assigns a vector or a set of vectors to this cluster by extending og appending to the class variable assigned_vectors.
        """
        vector_set_copy = copy.copy(vector_set)
        if isinstance(vector_set_copy[0], list):
            self.assigned_vectors.extend(vector_set_copy)
        else:
            self.assigned_vectors.append(vector_set_copy)
    
    def get_number_of_assigned_vectors(self):
        """
        Returns the number of assigned vectors in the cluster.
        """
        return len(self.assigned_vectors)
    
    def randomly_move_centre(self):
        random_seed = np.random.randint(-1, 1, len(self.mean))
        random_mean_multiplier = (random_seed/100) + 1
        self.mean = self.mean*random_mean_multiplier
        
    def is_empty(self):
        return self.get_number_of_assigned_vectors() == 0

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
        
