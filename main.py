from task_two_histogram import TaskTwoHistogram
from feature_structure import FeatureTranslationTable, GenreList
from knn import find_clusters, choose_reference_from_data, knn
from test import test_clustering
from audio_track import AudioTrackTable
        
def main():
    track_table = AudioTrackTable("GenreClassData_30s.txt")
    genre_lst = GenreList()
    feature_idxs = [
        FeatureTranslationTable.spectral_rolloff_mean.value,
        FeatureTranslationTable.mfcc_1_mean.value,
        FeatureTranslationTable.spectral_centroid_mean.value,
        FeatureTranslationTable.tempo.value
    ]
    clusters = choose_reference_from_data(track_table, genre_lst, feature_idxs)
    
    correct = 0
    wrong = 0
    test_tracks = track_table.get_test_set()
    
    for track in test_tracks:
        features = track.extract_features(feature_idxs)
        classified_genre = knn(features, clusters, 5)
        if classified_genre == track.genre:
            correct += 1
        else:
            wrong += 1
    error_rate = wrong/(correct + wrong)
    error_rate_percentage = error_rate*100
    print(error_rate_percentage)
    
    print("Correct", correct)
    print("Wrong:", wrong)
    
    '''for i in range(20):
        print(i)
        print()
        find_clusters()
        print()'''
    #test_clustering()
    
if __name__ == "__main__":
    main()
    
    
"""
Implement k Nearest Neighbour. This should input a vector and a set of clusters, then find the k nearest clusters from the vector, and use this to classify the vector into the correct genre.

Generate confusion matrix and error rate for the test set.

Fix the error with covariance and empty sets.
""" 