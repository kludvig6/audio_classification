from task_two_histogram import TaskTwoHistogram
from feature_structure import FeatureTranslationTable
from knn import find_clusters
from test import test_clustering

def main():
    find_clusters()
    #test_clustering()
    
if __name__ == "__main__":
    main()
    
    
"""
Implement k Nearest Neighbour. This should input a vector and a set of clusters, then find the k nearest clusters from the vector, and use this to classify the vector into the correct genre.

Generate confusion matrix and error rate for the test set.
""" 
