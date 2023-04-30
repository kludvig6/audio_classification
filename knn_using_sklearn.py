from audio_track import AudioTrackTable
from feature_structure import FeatureTranslationTable

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def knn_sklearn_classifier():
    audio_track_table = AudioTrackTable("GenreClassData_30s.txt")
    feature_idxs = [
        FeatureTranslationTable.spectral_rolloff_mean.value,
        FeatureTranslationTable.mfcc_1_mean.value,
        #FeatureTranslationTable.spectral_centroid_mean.value,
        FeatureTranslationTable.tempo.value
    ]
    
    training_set = audio_track_table.get_train_set()
    test_set = audio_track_table.get_test_set()
    
    X_train = [song.extract_features(feature_idxs) for song in training_set]
    y_train = [song.genre_id for song in training_set]
    
    X_test = [song.extract_features(feature_idxs) for song in test_set]
    y_test = [song.genre_id for song in test_set]
    
    knn = KNeighborsClassifier(n_neighbors=5)
    
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Error rate using sklearn:", 1-accuracy)