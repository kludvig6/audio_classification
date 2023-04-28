from __future__ import annotations
from audio_track import AudioTrackTable
import numpy as np
import matplotlib.pyplot as plt

def error_rate(true_lable_list):
    matrix = np.array(true_lable_list)
    true = 0
    wrong = 0
    
    i = 0
    while i < len(matrix):
        if matrix[i, 0] == matrix[i, 1]:
            true += 1
            i += 1
        else:
            wrong += 1
            i += 1
    
    return wrong / (true + wrong)

def error_rate_percentage(true_lable_list):
    matrix = np.array(true_lable_list)
    true = 0
    wrong = 0
    
    i = 0
    while i < len(matrix):
        if matrix[i, 0] == matrix[i, 1]:
            true += 1
            i += 1
        else:
            wrong += 1
            i += 1
    
    return (wrong / (true + wrong)) * 100
    
    
def confusion_matrix(true_lable_list):
    input_matrix = np.array(true_lable_list)
    confusion_zero = np.zeros((10, 10))
    true_values = []
    predicted_values = []
    
    
    i = 0
    for i in range(len(input_matrix)):
        true_values.append(input_matrix[i][0])
        predicted_values.append(input_matrix[i][1])
        
        
    j = 0
    while j < len(true_values):
        confusion_zero[true_values[j]][predicted_values[j]] += 1
        j += 1
    
    return confusion_zero

def plot_confusion_matrix(confusion_matrix):
    # plot confusion matrix
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix)

    # set axis labels
    ax.set_xticks(np.arange(len(confusion_matrix)))
    ax.set_yticks(np.arange(len(confusion_matrix)))
    ax.set_xticklabels(np.arange(10))
    ax.set_yticklabels(np.arange(10))
    
    # loop over data and add text annotations
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix)):
            text = ax.text(j, i, confusion_matrix[i, j],
                        ha="center", va="center", color="w")

    # add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # set title and show plot
    ax.set_title("Confusion Matrix")
    plt.show()