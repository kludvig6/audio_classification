import matplotlib.pyplot as plt

from audio_track import AudioTrackTable
from feature_structure import FeatureTranslationTable
from clustering import find_all_cluster_references

def test_clustering():
    track_table = AudioTrackTable("GenreClassData_30s.txt")
    pop_songs = track_table.get_specific_genre("pop")
    classical_songs = track_table.get_specific_genre("classical")
    
    #FeatureTranslationTable.spectral_rolloff_mean.value
    #FeatureTranslationTable.mfcc_1_mean.value
    #FeatureTranslationTable.spectral_centroid_mean.value
    #FeatureTranslationTable.tempo.value
    
    feature_idxs = [FeatureTranslationTable.spectral_rolloff_mean.value, FeatureTranslationTable.mfcc_1_mean.value]
    
    pop_features = [track.extract_features(feature_idxs) for track in pop_songs]
    classical_features = [track.extract_features(feature_idxs) for track in    classical_songs]
    
    pop_clusters = find_all_cluster_references(pop_features)
    pop_cluster_means = [cluster.mean for cluster in pop_clusters]
    
    classical_clusters = find_all_cluster_references(classical_features)
    classical_features_mean = [cluster.mean for cluster in classical_clusters]

    pop_x = [song_features[0] for song_features in pop_features]
    pop_y = [song_features[1] for song_features in pop_features]
    
    classical_x = [song_features[0] for song_features in classical_features]
    classical_y = [song_features[1] for song_features in classical_features]
    
    pop_z = [mean[0] for mean in pop_cluster_means]
    pop_w = [mean[1] for mean in pop_cluster_means]
    
    classical_z = [mean[0] for mean in classical_features_mean]
    classical_w = [mean[1] for mean in classical_features_mean]
    
    plt.scatter(pop_x, pop_y, marker=".", c="blue", label="pop features")
    plt.scatter(classical_x, classical_y, marker=".", c="green", label="classical features")
    plt.scatter(pop_z, pop_w, marker="x", c="purple", label="pop_clusters")
    plt.scatter(classical_z, classical_w, marker="x", c="red", label="classical clusters")
    plt.legend(loc="lower right")
    
    plt.xlabel("Spectral rolloff mean")
    plt.ylabel("Mfcc 1 mean")
    plt.title("Test of clustering algorithm")
    plt.show()