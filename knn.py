import numpy as np

from utils import euclidian

GENRE_TO_NUMBER = {"pop": 0, "metal": 1, "disco": 2, "blues": 3, "reggae":  4, "classical": 5, "rock": 6, "hiphop": 7, "country": 8, "jazz": 9}

NUMBER_TO_GENRE = {0: "pop", 1: "metal", 2: "disco", 3: "blues", 4: "reggae", 5: "classical", 6: "rock", 7: "hiphop", 8: "country", 9: "jazz"}

def knn(song, training_set, feature_idxs, k):
    """
    This function classifies a vector x to the correct class using the k-NN method. The whole training set is used as potential neighbours, and k indicates the number of neighbours to find.
    """
    test_features = song.extract_features(feature_idxs)
    training_set_features = [song.extract_features(feature_idxs) for song in training_set]
     
    distances = [euclidian(test_features, train_features) for train_features in training_set_features]
    
    k_smallest_distance = kmin(distances, k)
    
    nearest_neighbours = []
    for dist in k_smallest_distance:
        song_idx = distances.index(dist)
        nearest_neighbours.append(training_set[song_idx])
        
    genre_occurence_count = np.zeros(10)
    for song in nearest_neighbours:
        genre_idx = GENRE_TO_NUMBER[song.genre]
        genre_occurence_count[genre_idx] += 1
    
    max_genre_occurence = max(genre_occurence_count)
    max_occurence_indices = [idx for idx, genre_occurence in enumerate(genre_occurence_count) if genre_occurence == max_genre_occurence]

    # If only one max, choose this genre
    if len(max_occurence_indices) == 1:
        classified_genre = NUMBER_TO_GENRE[max_occurence_indices[0]]
    # If tie between genres, choose nearest
    else:
        max_genres = [NUMBER_TO_GENRE[idx] for idx in max_occurence_indices]
        for song in nearest_neighbours:
            if song.genre in max_genres:
                classified_genre = song.genre
                break
    
    return classified_genre
    
def kmin(values, k):
    """
    Returns the k lowest values from a list of values.
    """
    return sorted(values)[:k]     
    