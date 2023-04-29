from task_two_histogram import TaskTwoHistogram
from feature_structure import FeatureTranslationTable, GenreList
from knn import knn
from error_rate import error_rate, confusion_matrix, error_rate_percentage, plot_confusion_matrix, specific_error_rates
from audio_track import AudioTrackTable

import numpy as np

import torch
from torch import nn
from neural_network import NeuralNetwork

#def main():
#    device = (
#    "cuda"
#    if torch.cuda.is_available()
#    else "mps"
#    if torch.backends.mps.is_available()
#    else "cpu"
#    )
#    print(f"Using {device} device")
#
#    model = NeuralNetwork().to(device)
#    print(model)
#    
#    X = torch.rand(1, 28, 28, device=device)
#    logits = model(X)
#    pred_probab = nn.Softmax(dim=1)(logits)
#    y_pred = pred_probab.argmax(1)
#    print(f"Predicted class: {y_pred}")
#    
#    """ Below here a image is used on the model. """
#    input_image = torch.rand(3,28,28)
#    print(input_image.size())
#    flatten = nn.Flatten()
#    flat_image = flatten(input_image)
#    print(flat_image.size())
#    
#    """ Linear - linear transformation layer"""
#    layer1 = nn.Linear(in_features=28*28, out_features=20)
#    hidden1 = layer1(flat_image)
#    print(hidden1.size())
#    
#    """ - ReLU - non-linear activation to help network learn non-linearity """
#    print(f"Before ReLU: {hidden1}\n\n")
#    hidden1 = nn.ReLU()(hidden1)
#    print(f"After ReLU: {hidden1}")
#    
#    """ Sequential - ordered container of modules"""
#    seq_modules = nn.Sequential(
#    flatten,
#    layer1,
#    nn.ReLU(),
#    nn.Linear(20, 10)
#    )
#    input_image = torch.rand(3,28,28)
#    logits = seq_modules(input_image)
#    print("Logits before softmax:", logits)
#    
#    """ Softmax - scale output probabilities to [0, 1] """
#    softmax = nn.Softmax(dim=1)
#    pred_probab = softmax(logits)
#    print("Probabilities after softmax", pred_probab)
#    
#    """ Model parameters - the networks weights and biases """
#    print(f"Model structure: {model}\n\n")
#    for name, param in model.named_parameters():
#        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
#        

NUMBER_OF_NEIGHBOURS = 5

def main():
    audio_track_table = AudioTrackTable("GenreClassData_30s.txt")
    genre_lst = GenreList()
    feature_idxs = [
        FeatureTranslationTable.spectral_rolloff_mean.value,
        FeatureTranslationTable.mfcc_1_mean.value,
        FeatureTranslationTable.spectral_centroid_mean.value,
        FeatureTranslationTable.tempo.value
    ]
    
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
    print("Error rate per genre: ", err_rate_per_genre)
    print("Confusion matrix:\n", conf_matrix)
    
    plot_confusion_matrix(conf_matrix)
    
if __name__ == "__main__":
    main()
    
    
"""
Generate confusion matrix and error rate for the test set.

Fix the error with covariance and empty sets.

Write a neural network for task 4.
""" 
