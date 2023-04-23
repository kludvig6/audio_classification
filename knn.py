import matplotlib.pyplot as plt

from clustering import find_five_clusters
from utils import mahalanobis
from audio_track import AudioTrackTable
from feature_structure import GenreList, FeatureTranslationTable

def knn(x, clusters, k):
    
    distances = [mahalanobis(x, cluster) for cluster in clusters]
    k_smallest_distance = kmin(distances, k)
    
    nearest_clusters = []
    for dist in k_smallest_distance:
        nearest_clusters.append(clusters[distances.index(dist)])
    
    for cluster in nearest_clusters:
        #count the number of genres
        print()

"""FeatureTranslationTable.tempo.value,
FeatureTranslationTable.mfcc_1_mean.value,
FeatureTranslationTable.spectral_centroid_mean.value,"""
def find_clusters():
    track_table = AudioTrackTable("GenreClassData_30s.txt")
    genre_lst = GenreList()
    feature_idxs = [
        FeatureTranslationTable.spectral_rolloff_mean.value,      
        FeatureTranslationTable.mfcc_1_mean.value
        ]
    all_clusters = []
    
    for genre in genre_lst.genres:
        songs = track_table.get_specific_genre(genre)
        features = [track.extract_features(feature_idxs) for track in songs]
        clusters = find_five_clusters(features, genre)
        all_clusters.append(clusters)
    
 
    for clusters in all_clusters:
        print(clusters[0].genre)
        for cluster in clusters:
            print(cluster.mean)
    
    
    
def kmin(values, k):
    return sorted(values)[:k]     
    