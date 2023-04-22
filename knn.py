from utils import mahalanobis

def knn(x, clusters, k):
    
    distances = [mahalanobis(x, cluster) for cluster in clusters]
    k_smallest_distance = kmin(distances, k)
    
    nearest_clusters = []
    for dist in k_smallest_distance:
        nearest_clusters.append(clusters[distances.index(dist)])
    
    for cluster in nearest_clusters:
        #count the number of genres
        
    
    
def kmin(values, k):
    return sorted(values)[:k]     
    