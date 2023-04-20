import numpy as np

from feature_structure import FeatureTranslationTable,GenreTable

"""
Old dataloader using a 2D-list instead of class.
class DataLoader:
    def __init__(self, _file): 
        filename = "Classification music/data/" + _file
        data_file = open(filename, "r")
        
        self.data = []
        for entry in data_file:
            self.data.append(entry.split())
        self.data = np.array(self.data[1:])

    def get_specific_genre(self, genre):
        mask = self.data[:,FeatureTranslationTable.Genre.value] == genre
        return self.data[mask]
"""