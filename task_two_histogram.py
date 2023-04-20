import matplotlib.pyplot as plt

from audio_track import AudioTrackTable

class TaskTwoHistogram:
    def __init__(self):
        self.track_table = AudioTrackTable("GenreClassData_30s.txt")
        self.pop_songs = self.track_table.get_specific_genre("pop")
        self.disco_songs = self.track_table.get_specific_genre("disco")
        self.metal_songs = self.track_table.get_specific_genre("metal")
        self.classical_songs = self.track_table.get_specific_genre("classical")
    
    def create_overlapping_histogram(self, feature):
        """
        Inputs an enum from FeatureTranslationTable and creates histogram for pop, disco, metal and classical songs for the given feature.
        """
        pop = [song.features[feature.value] for song in self.pop_songs]
        disco = [song.features[feature.value] for song in self.disco_songs]
        metal = [song.features[feature.value] for song in self.metal_songs]
        classical = [song.features[feature.value] for song in self.classical_songs]
        
        plt.hist(pop, color="blue", bins=32, label="Pop")
        plt.hist(disco, color="green", bins=32, label="Disco")
        plt.hist(metal, color="red", bins=32, label="Metal")
        plt.hist(classical, color="purple", bins=32, label="Classical")

        plt.legend(loc='upper right')
        plt.title(feature.name)
        plt.show()       
        
    def create_histogram(self, feature):
        """
        Inputs an enum from FeatureTranslationTable and creates subplots with histogram of pop, disco, metal and classical songs for the given feature.
        """
        pop = [song.features[feature.value] for song in self.pop_songs]
        disco = [song.features[feature.value] for song in self.disco_songs]
        metal = [song.features[feature.value] for song in self.metal_songs]
        classical = [song.features[feature.value] for song in self.classical_songs]
        
        plt.subplot(2,2,1)
        plt.hist(pop, color="blue", bins=16)
        plt.title("Pop")
        
        plt.subplot(2,2,2)
        plt.hist(disco, color="green", bins=16)
        plt.title("Disco")
        
        plt.subplot(2,2,3)
        plt.hist(metal, color="red", bins=16)
        plt.title("Metal")
        
        plt.subplot(2,2,4)
        plt.hist(classical, color="purple", bins=16)
        plt.title("Classical")
        
        plt.suptitle(feature.name)
        plt.show()