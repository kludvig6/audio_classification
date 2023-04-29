import numpy as np
import random

from deprecated_code.cluster import Cluster
from deprecated_code.clustering import new_find_five_clusters
from utils import mahalanobis
from audio_track import AudioTrackTable
from feature_structure import GenreList, FeatureTranslationTable

GENRE_TO_NUMBER = {"pop": 0, "metal": 1, "disco": 2, "blues": 3, "reggae":  4, "classical": 5, "rock": 6, "hiphop": 7, "country": 8, "jazz": 9}

NUMBER_TO_GENRE = {0: "pop", 1: "metal", 2: "disco", 3: "blues", 4: "reggae", 5: "classical", 6: "rock", 7: "hiphop", 8: "country", 9: "jazz"}

def knn(x, clusters, k):
    """
    This function classifies a vector x to the correct class using the k-NN method with the given clusters.
    """
    distances = [mahalanobis(x, cluster) for cluster in clusters]
    k_smallest_distance = kmin(distances, k)
    
    nearest_clusters = []
    for dist in k_smallest_distance:
        nearest_clusters.append(clusters[distances.index(dist)])
    
    genres = np.zeros(10)
    for cluster in nearest_clusters:
        genre_idx = GENRE_TO_NUMBER[cluster.genre]
        genres[genre_idx] += 1
    
    max_genre_occurence = max(genres)
    max_genre_indices = [idx for idx, genre_occurence in enumerate(genres) if genre_occurence == max_genre_occurence]
    
    if len(max_genre_indices) == 1:
        classified_genre = NUMBER_TO_GENRE[max_genre_indices[0]]
    else:
        max_genres = [NUMBER_TO_GENRE[idx] for idx in max_genre_indices]
        for cluster in nearest_clusters:
            if cluster.genre in max_genres:
                classified_genre = cluster.genre
                break
    
    return classified_genre

def choose_reference_from_data(track_table, genre_lst, feature_idxs):    
    all_clusters = []
    for genre in genre_lst.genres:
        songs = track_table.get_specific_genre(genre)
        references = random.choices(songs, k=5)
        clusters = []
        for ref in references:
            features = ref.extract_features(feature_idxs)
            cluster = Cluster(features, np.eye(len(feature_idxs)), genre)
            clusters.append(cluster)
        all_clusters.append(clusters)
    

    all_clusters = [c for clusters in all_clusters for c in clusters]

    return all_clusters
    

def find_clusters():
    """
    Finds five clusters for each genre.
    """
    track_table = AudioTrackTable("GenreClassData_30s.txt")
    genre_lst = GenreList()
    feature_idxs = [
        FeatureTranslationTable.spectral_rolloff_mean.value,
        FeatureTranslationTable.mfcc_1_mean.value,
        FeatureTranslationTable.spectral_centroid_mean.value,
        FeatureTranslationTable.tempo.value
    ]
    all_clusters = []
    
    for genre in genre_lst.genres:
        songs = track_table.get_specific_genre(genre)
        features = [track.extract_features(feature_idxs) for track in songs]
        clusters = new_find_five_clusters(features, genre)
        all_clusters.append(clusters)
    
 
    for clusters in all_clusters:
        print(clusters[0].genre)
        for cluster in clusters:
            print(cluster.mean)
            
            
def kmin(values, k):
    """
    Returns the k lowest values from a list of values.
    """
    return sorted(values)[:k]  