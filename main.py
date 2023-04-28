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
    test_tracks = track_table.get_test_set()
    
    classification_results = []
    for track in test_tracks:
        features = track.extract_features(feature_idxs)
        classified_genre = knn(features, clusters, 5)
        classified_genre_id = genre_lst.genres.index(classified_genre)
        true_genre_id = track.genre_id
        classification_results.append([true_genre_id, classified_genre_id])
        
    wrong = 0
    right = 0
    for true_label_pair in classification_results:
        if true_label_pair[0] == true_label_pair[1]:
            right += 1
        else:
            wrong += 1
    print("Right:", right)
    print("Wrong:", wrong)
    print("Error rate:", wrong/(right+wrong))
    
    '''for i in range(20):
        print(i)
        print()
        find_clusters()
        print()'''
    
if __name__ == "__main__":
    main()
    
    
"""
Generate confusion matrix and error rate for the test set.

Fix the error with covariance and empty sets.

Write a neural network for task 4.
""" 