import pandas as pd
from scipy.optimize import linear_sum_assignment

def localize_drift(old_centroids, new_centroids, col_names):
    if isinstance(old_centroids, pd.Series):
        old_centroids = old_centroids.mean(axis=0)
    
    if isinstance(new_centroids, pd.Series):
        new_centroids = new_centroids.mean(axis=0)
        
    r, c = linear_sum_assignment(distance.cdist(old_centroids, new_centroids))
    lut = c
    
    resp = pd.DataFrame([
        ((old_centroids - new_centroids[lut]) ** 2).mean(axis=0)
    ], columns=col_names).transpose()
    
    return resp