from __future__ import annotations
from audio_track import AudioTrackTable

def error_rate():
    test_track_table = AudioTrackTable("GenreClassData_30s.txt")
    type_tests = []
    type_tests.extend(test_track_table.get_tests())     #Length  of this vector is number of tests
    
    i = 0
    #while i < len(type_tests):