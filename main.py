from task_two_histogram import TaskTwoHistogram
from feature_structure import FeatureTranslationTable, GenreList
from knn import knn
from error_rate import error_rate, confusion_matrix, error_rate_percentage, plot_confusion_matrix, specific_error_rates
from audio_track import AudioTrackTable
from knn_using_sklearn import knn_sklearn_classifier
import numpy as np



import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from neural_network import NeuralNetwork, train, test, one_hot_encoder


BATCH_SIZE = 100
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

TASK_3_FEATURES = [
    FeatureTranslationTable.spectral_rolloff_mean.value,
    FeatureTranslationTable.mfcc_1_mean.value,
    FeatureTranslationTable.spectral_centroid_mean.value,
]

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
    
    feature_idxs = range(63)
    
    X_train = [song.extract_features(feature_idxs) for song in training_set]
    y_train = [song.genre_id for song in training_set]
    y_train = [one_hot_encoder(y) for y in y_train]
    
    X_test = [song.extract_features(feature_idxs) for song in test_set]
    y_test = [song.genre_id for song in test_set]
    #y_test = [one_hot_encoder(y) for y in y_test]
    
    x_train_loader = DataLoader(X_train, batch_size=BATCH_SIZE, drop_last=True)
    y_train_loader = DataLoader(y_train, batch_size=BATCH_SIZE, drop_last=True)

    x_test_loader = DataLoader(X_test, batch_size=1, drop_last=True)
    y_test_loader = DataLoader(y_test, batch_size=1, drop_last=True)
    
    
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(model)
    
    train(model, x_train_loader, y_train_loader, optimizer, device)
    
    classification_results = test(model, x_test_loader, y_test_loader, device)

    err_rate = error_rate(classification_results)
    conf_matrix = confusion_matrix(classification_results)
    err_rate_per_genre = specific_error_rates(conf_matrix)
    
    print("Error rate:", err_rate)
    #print("Error rate per genre: ", err_rate_per_genre)
    print("Confusion matrix:\n", conf_matrix)
    
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
    
    err_rate = error_rate(classification_results)
    conf_matrix = confusion_matrix(classification_results)
    err_rate_per_genre = specific_error_rates(conf_matrix)
    
    print("Error rate:", err_rate)
    #print("Error rate per genre: ", err_rate_per_genre)
    print("Confusion matrix:\n", conf_matrix)
    
    #plot_confusion_matrix(conf_matrix)
    
if __name__ == "__main__":
    main()
    neural_network_main()
    
    
"""
Generate confusion matrix and error rate for the test set.

Fix the error with covariance and empty sets.

Write a neural network for task 4.
""" 
