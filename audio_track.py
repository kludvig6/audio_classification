import numpy as np

class AudioTrack:
    def __init__(self, _file_entry):
        audio_track_metadata = _file_entry.split()
        
        self.track_id = int(audio_track_metadata[0])
        self.file = audio_track_metadata[1]
        self.features = np.array([float(x) for x in audio_track_metadata[2:65]])
        self.genre_id = int(audio_track_metadata[65])
        self.genre = audio_track_metadata[66]
        self.type = audio_track_metadata[67]

        self.cluster = None
        
    def __str__(self):
        return str([self.track_id, self.file, self.features, self.genre_id, self.genre, self.type])
    
class AudioTrackTable:
    def __init__(self, _filename):
        filename = "data/" + _filename
        data_file = open(filename, "r")
        next(data_file)
        
        self.track_table = []
        for entry in data_file:
            self.track_table.append(AudioTrack(entry))
    
    def get_specific_genre(self, genre):
        return [track for track in self.track_table if track.genre == genre]