from __future__ import annotations
from audio_track import AudioTrackTable
import numpy as np

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
    for i in input_matrix:
        true_values.append[input_matrix[i][0]]
        predicted_values.append[input_matrix[i][1]]
    
    j = 0
    while j < len(true_values):
        confusion_zero[true_values[j]][predicted_values[j]] += 1
    
    return confusion_zero