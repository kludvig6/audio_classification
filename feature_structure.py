"""
FeatureTranslationTable containts the indices for the features in AudioTrack class.
"""

from enum import Enum

class FeatureTranslationTable(Enum):
    zero_cross_rate_mean    = 0  
    zero_cross_rate_std     = 1  
    rmse_mean               = 2  
    rmse_var                = 3  
    spectral_centroid_mean  = 4  
    spectral_centroid_var   = 5   
    spectral_bandwidth_mean = 6  
    spectral_bandwidth_var  = 7  
    spectral_rolloff_mean   = 8 
    spectral_rolloff_var    = 9 
    spectral_contrast_mean  = 10
    spectral_contrast_var   = 11
    spectral_flatness_mean  = 12 
    spectral_flatness_var   = 13 
    chroma_stft_1_mean      = 14 
    chroma_stft_2_mean      = 15 
    chroma_stft_3_mean      = 16 
    chroma_stft_4_mean      = 17 
    chroma_stft_5_mean      = 18 
    chroma_stft_6_mean      = 19 
    chroma_stft_7_mean      = 20 
    chroma_stft_8_mean      = 21 
    chroma_stft_9_mean      = 22 
    chroma_stft_10_mean     = 23 
    chroma_stft_11_mean     = 24 
    chroma_stft_12_mean     = 25 
    chroma_stft_1_std       = 26 
    chroma_stft_2_std       = 27 
    chroma_stft_3_std       = 28 
    chroma_stft_4_std       = 29 
    chroma_stft_5_std       = 30 
    chroma_stft_6_std       = 31 
    chroma_stft_7_std       = 32 
    chroma_stft_8_std       = 33 
    chroma_stft_9_std       = 34 
    chroma_stft_10_std      = 35 
    chroma_stft_11_std      = 36 
    chroma_stft_12_std      = 37 
    tempo                   = 38 
    mfcc_1_mean             = 39 
    mfcc_2_mean             = 40 
    mfcc_3_mean             = 41 
    mfcc_4_mean             = 42 
    mfcc_5_mean             = 43 
    mfcc_6_mean             = 44 
    mfcc_7_mean             = 45 
    mfcc_8_mean             = 46 
    mfcc_9_mean             = 47 
    mfcc_10_mean            = 48 
    mfcc_11_mean            = 49  
    mfcc_12_mean            = 50  
    mfcc_1_std              = 51  
    mfcc_2_std              = 52 
    mfcc_3_std              = 53 
    mfcc_4_std              = 54 
    mfcc_5_std              = 55 
    mfcc_6_std              = 56 
    mfcc_7_std              = 57 
    mfcc_8_std              = 58 
    mfcc_9_std              = 59 
    mfcc_10_std             = 60 
    mfcc_11_std             = 61 
    mfcc_12_std             = 62

class GenreList():
    def __init__(self):
        self.genres =  ["pop", "metal", "disco", "blues", "reggae", "classical", "rock", "hiphop", "country", "jazz"]
        
class GenreToNumber:
    def __init__(self):
        self.GenreTranslationTable = {"pop": 1, "metal": 2, "disco": 3, "blues": 4, "reggae":  5, "classical": 6, "rock": 7, "hiphop": 8, "country": 9, "jazz": 10}
        
class NumberToGenre:
    def __init__(self):
        self.GenreTranslationTable = {1: "pop", 2: "metal", 3: "disco", 4: "blues", 5: "reggae", 6: "classical", 7: "rock", 8: "hiphop", 9: "country", 10: "jazz"}