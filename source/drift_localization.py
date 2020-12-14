import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

def localize_drift(before_drift_centroids, after_drift_centroids, col_names):
    """
        Performs the drift localization method based on the MSE 
        between centroids from the windows before and after the drift.

        The centroid matching step is solved using the linear sum assignment
        Hungarian method proposed in `Harold W. Kuhn. The Hungarian Method for the
        assignment problem. Naval Research Logistics Quarterly, 2:83-97, 1955`

        Parameters:
            before_drift_centroids (pd.DataFrame): Centroids before the drift
            after_drift_centroids (pd.DataFrame): Centroids after the drift
            col_names (list): List with the names of the columns relating to 
                each dimension of the traces vector representation (activities 
                or transitions)
            
        Returns:
            pd.DataFrame: DataFrame with the MSE of each dimension 
    """
    if isinstance(before_drift_centroids, pd.Series):
        before_drift_centroids = before_drift_centroids.mean(axis=0)
    
    if isinstance(after_drift_centroids, pd.Series):
        after_drift_centroids = after_drift_centroids.mean(axis=0)

    _ , lut = linear_sum_assignment(distance.cdist(before_drift_centroids, after_drift_centroids))
    
    resp = pd.DataFrame([
        ((before_drift_centroids - after_drift_centroids[lut]) ** 2).mean(axis=0)
    ], columns=col_names).transpose().sort_values(0, ascending=False)
    
    return resp


def localize_all_drifts(run_df, drifts, cluster_window_size, col_names):
    drifts_localization = []

    for i in range(len(drifts)):
        drifts_localization.append(
            localize_drift(
                run_df.centroids.loc[drifts[i] - cluster_window_size], 
                run_df.centroids.loc[drifts[i]], 
                col_names
            ).rename(columns={0: 'drift_'+str(drifts[i])})
        )
    
    return pd.concat(drifts_localization, axis=1)