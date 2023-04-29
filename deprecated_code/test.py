import matplotlib.pyplot as plt

from audio_track import AudioTrackTable
from feature_structure import FeatureTranslationTable
from deprecated_code.clustering import find_five_clusters
from utils import mahalanobis_all_clusters

def test_clustering():
    track_table = AudioTrackTable("GenreClassData_30s.txt")
    disco_songs = track_table.get_specific_genre("disco")
    metal_songs = track_table.get_specific_genre("metal")
    
    #FeatureTranslationTable.spectral_rolloff_mean.value
    #FeatureTranslationTable.mfcc_1_mean.value
    #FeatureTranslationTable.spectral_centroid_mean.value
    #FeatureTranslationTable.tempo.value
    
    feature_idxs = [FeatureTranslationTable.spectral_rolloff_mean.value, FeatureTranslationTable.mfcc_1_mean.value]
    
    disco_features = [track.extract_features(feature_idxs) for track in disco_songs]
    metal_features = [track.extract_features(feature_idxs) for track in    metal_songs]
    
    disco_clusters = find_five_clusters(disco_features, "disco")
    disco_cluster_means = [cluster.mean for cluster in disco_clusters]
    
    metal_clusters = find_five_clusters(metal_features, "metal")
    metal_features_mean = [cluster.mean for cluster in metal_clusters]

    disco_x = [song_features[0] for song_features in disco_features]
    disco_y = [song_features[1] for song_features in disco_features]
    
    metal_x = [song_features[0] for song_features in metal_features]
    metal_y = [song_features[1] for song_features in metal_features]
    
    disco_z = [mean[0] for mean in disco_cluster_means]
    disco_w = [mean[1] for mean in disco_cluster_means]
    
    metal_z = [mean[0] for mean in metal_features_mean]
    metal_w = [mean[1] for mean in metal_features_mean]
    
    disco_mah = mahalanobis_all_clusters(disco_features, disco_clusters)
    metal_mah = mahalanobis_all_clusters(metal_features, metal_clusters)
    
    print("Disco mahalnobis:", disco_mah)
    print("Metal mahalanobis:", metal_mah)
    
    plt.scatter(disco_x, disco_y, marker=".", c="blue", label="disco features")
    plt.scatter(metal_x, metal_y, marker=".", c="green", label="metal features")
    plt.scatter(disco_z, disco_w, marker="x", c="purple", label="disco clusters")
    plt.scatter(metal_z, metal_w, marker="x", c="red", label="metal clusters")
    plt.legend(loc="lower right")
    
    plt.xlabel("Spectral rolloff mean")
    plt.ylabel("Mfcc 1 mean")
    plt.title("Test of clustering algorithm")
    plt.show()