from task_two_histogram import TaskTwoHistogram
from feature_structure import FeatureTranslationTable, GenreList
from knn import find_clusters, choose_reference_from_data, knn
from test import test_clustering
from error_rate import error_rate
from error_rate import confusion_matrix
from error_rate import error_rate_percentage
from error_rate import plot_confusion_matrix



from audio_track import AudioTrackTable

import torch
from torch import nn
from neural_network import NeuralNetwork

def main():
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    print(model)
    
    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
    
    """ Below here a image is used on the model. """
    input_image = torch.rand(3,28,28)
    print(input_image.size())
    flatten = nn.Flatten()
    flat_image = flatten(input_image)
    print(flat_image.size())
    
    """ Linear - linear transformation layer"""
    layer1 = nn.Linear(in_features=28*28, out_features=20)
    hidden1 = layer1(flat_image)
    print(hidden1.size())
    
    """ - ReLU - non-linear activation to help network learn non-linearity """
    print(f"Before ReLU: {hidden1}\n\n")
    hidden1 = nn.ReLU()(hidden1)
    print(f"After ReLU: {hidden1}")
    
    """ Sequential - ordered container of modules"""
    seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
    )
    input_image = torch.rand(3,28,28)
    logits = seq_modules(input_image)
    print("Logits before softmax:", logits)
    
    """ Softmax - scale output probabilities to [0, 1] """
    softmax = nn.Softmax(dim=1)
    pred_probab = softmax(logits)
    print("Probabilities after softmax", pred_probab)
    
    """ Model parameters - the networks weights and biases """
    print(f"Model structure: {model}\n\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        
"""def main():
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
    print("Error rate:", wrong/(right+wrong))"""
    
    #print()
    
    #print("error rate:", error_rate_percentage(classification_results))
    #print("confusion matrix:")
    #print(confusion_matrix(classification_results))
    #print(classification_results)
    #plot_confusion_matrix(confusion_matrix(classification_results))
    


    
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
