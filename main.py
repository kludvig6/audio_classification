from task_two_histogram import TaskTwoHistogram
from feature_structure import FeatureTranslationTable, GenreList
from knn import knn
from error_rate import error_rate, confusion_matrix, error_rate_percentage, plot_confusion_matrix, specific_error_rates
from audio_track import AudioTrackTable
from knn_using_sklearn import knn_sklearn_classifier
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader
from neural_network import NeuralNetwork, train, test, one_hot_encoder

BATCH_SIZE = 700
NUM_FEATURES = 63
NUMBER_OF_NEIGHBOURS = 5
TASK_NUMBER = 1

TASK_1_FEATURES = [
    FeatureTranslationTable.spectral_rolloff_mean.value,
    FeatureTranslationTable.mfcc_1_mean.value,
    FeatureTranslationTable.spectral_centroid_mean.value,
    FeatureTranslationTable.tempo.value
]

TASK_2_FEATURES = [
    FeatureTranslationTable.spectral_rolloff_mean.value,
    FeatureTranslationTable.mfcc_1_mean.value,
    FeatureTranslationTable.spectral_centroid_mean.value,
]

TASK_2_FEATURES_TESTS = [
    FeatureTranslationTable.spectral_rolloff_mean.value,
    FeatureTranslationTable.mfcc_1_mean.value,
    FeatureTranslationTable.tempo.value,
    #FeatureTranslationTable.spectral_centroid_mean.value,
]

TASK_3_FEATURES = [
    FeatureTranslationTable.spectral_rolloff_mean.value,
    FeatureTranslationTable.mfcc_1_mean.value,
    FeatureTranslationTable.spectral_centroid_mean.value,
]

TASK_4_FEATURES = range(NUM_FEATURES)

def neural_network_main():
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    audio_track_table = AudioTrackTable("GenreClassData_30s.txt")
    training_set = audio_track_table.get_train_set()
    test_set = audio_track_table.get_test_set()
    
    feature_idxs = TASK_4_FEATURES
    
    scaler = StandardScaler()
    
    X_train = [song.extract_features(feature_idxs) for song in training_set]
    X_train = scaler.fit_transform(X_train)
    
    y_train = [song.genre_id for song in training_set]
    y_train = [one_hot_encoder(y) for y in y_train]
    
    X_test = [song.extract_features(feature_idxs) for song in test_set]
    X_test = scaler.fit_transform(X_test)
    
    y_test = [song.genre_id for song in test_set]
        
    x_train_loader = DataLoader(X_train, batch_size=len(X_train), drop_last=True)
    y_train_loader = DataLoader(y_train, batch_size=len(X_train), drop_last=True)

    x_test_loader = DataLoader(X_test, batch_size=1, drop_last=True)
    y_test_loader = DataLoader(y_test, batch_size=1, drop_last=True)
    
    model = NeuralNetwork(len(feature_idxs)).to(device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
    
    train(model, x_train_loader, y_train_loader, optimizer, device)
    
    classification_results = test(model, x_test_loader, y_test_loader, device)
    for result in classification_results:
        result[0] = int(result[0])
        
    err_rate = error_rate(classification_results)
    conf_matrix = confusion_matrix(classification_results)
    
    print("Error rate:", err_rate)
    print("Confusion matrix:\n", conf_matrix)

def main_find_lowest_error_rate():
    audio_track_table = AudioTrackTable("GenreClassData_30s.txt")
    genre_lst = GenreList()
    
    feature_idxs_baseline = TASK_2_FEATURES_TESTS
    
    lowest_error_rate = 1
    lowest_error_rate_idx = 0
    for i in range(63):
        if i in feature_idxs_baseline:
            pass
        feature_idxs = feature_idxs_baseline + [i]
        
        training_set = audio_track_table.get_train_set()
        test_set = audio_track_table.get_test_set()

        classification_results = []

        for song in test_set:
            classified_genre = knn(song, training_set, feature_idxs, NUMBER_OF_NEIGHBOURS)
            classified_genre_id = genre_lst.genres.index(classified_genre)
            true_genre_id = song.genre_id
            classification_results.append([true_genre_id, classified_genre_id]) 

        for result in classification_results:
            result[0] = int(result[0])

        err_rate = error_rate(classification_results)
        conf_matrix = confusion_matrix(classification_results)
    
        print("Error rate:", err_rate)
        print("Confusion matrix:\n", conf_matrix)
        
        if err_rate < lowest_error_rate:
            lowest_error_rate = err_rate
            lowest_error_rate_idx = i
    print(f"Lowest error rate is {lowest_error_rate} for feature {lowest_error_rate_idx}")
        
def main():
    knn_sklearn_classifier()
    
    audio_track_table = AudioTrackTable("GenreClassData_30s.txt")
    genre_lst = GenreList()

    if TASK_NUMBER == 1:
        feature_idxs = TASK_1_FEATURES
    elif TASK_NUMBER == 2:
        feature_idxs = TASK_2_FEATURES
    elif TASK_NUMBER == 3:
        feature_idxs = TASK_3_FEATURES
    
    training_set = audio_track_table.get_train_set()
    test_set = audio_track_table.get_test_set()
    
    classification_results = []
    
    for song in test_set:
        classified_genre = knn(song, training_set, feature_idxs, NUMBER_OF_NEIGHBOURS)
        classified_genre_id = genre_lst.genres.index(classified_genre)
        true_genre_id = song.genre_id
        classification_results.append([true_genre_id, classified_genre_id]) 
    
    for result in classification_results:
        result[0] = int(result[0])
        
    err_rate = error_rate(classification_results)
    conf_matrix = confusion_matrix(classification_results)
    
    print("Error rate:", err_rate)
    print("Confusion matrix:\n", conf_matrix)
    
    plot_confusion_matrix(conf_matrix)
    
if __name__ == "__main__":
    #main_find_lowest_error_rate()
    #neural_network_main()
    main()
