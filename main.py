from task_two_histogram import TaskTwoHistogram
from feature_structure import FeatureTranslationTable

def main():    
    task = TaskTwoHistogram()
    
    task.create_overlapping_histogram(FeatureTranslationTable.spectral_rolloff_mean)
    task.create_overlapping_histogram(FeatureTranslationTable.mfcc_1_mean)
    task.create_overlapping_histogram(FeatureTranslationTable.spectral_centroid_mean)
    task.create_overlapping_histogram(FeatureTranslationTable.tempo)

if __name__ == "__main__"
    main()