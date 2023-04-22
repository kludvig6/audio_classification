import matplotlib.pyplot as plt
import random
import numpy as np

from task_two_histogram import TaskTwoHistogram
from feature_structure import FeatureTranslationTable
from cluster import Cluster
from audio_track import AudioTrackTable

from test import test_clustering

def main():
    
    a = [ 4, 2, 7, 8, 5, 9]
    a_min = sorted(a)[:3]
    
    minimum = []
    for min in a_min:
        minimum.append(a[a.index(min)])
    print(minimum)
    test_clustering()
    
if __name__ == "__main__":
    main()
    
    
"""
Next to implement: The clusters does not have genres given to them. This should be implemented. The clustering algorithm is per class, but when implementing the nearest neighbour, class will be needed.

How many cluster should be implemented. Should there be five clusters per class, or should the clustering algorithm be able to decide?

Implement k Nearest Neighbour. This should input a vector and a set of clusters, then find the k nearest clusters from the vector, and use this to classify the vector into the correct genre.

At the start of main, create the AudioTrackTable, and then create the clusters for each class using the train-tracks. Then, loop through the test-tracks and classify each of them using k Nearest Neighbour.

Generate confusion matrix and error rate for the test set.
""" 