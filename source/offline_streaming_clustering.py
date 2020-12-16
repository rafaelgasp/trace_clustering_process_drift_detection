from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error
from scipy.spatial import distance
from scipy.stats import skew
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from copy import deepcopy
from sklearn.base import clone as sk_clone


def get_validation_indexes(X, y_pred):
    """
        Returns clustering validation indexes (Silhouette and DBi) 
        based on input X and groups y_pred.
    """
    return {
        "Silhouette": silhouette_score(X, y_pred),
        "DBi": davies_bouldin_score(X, y_pred),
    }


def get_centroids_metrics(X, y_pred, centroids):
    """
        Calculate trace clustering features (Radius, intra-cluster 
        distance, skewness and standard deviation for each cluster)
        based on centroids and traces. 

        Parameters:
        -----------
            X (pd.DataFrame): Dataset of traces in vector space representation
            y_pred (np.array): Attribution of each trace to one cluster
            centroids (np.array): Centroids of each cluster
    """
    r = {
        "radius_list": [],
        "dist_intra_cluster_list": [],
        "skewness_list": [],
        "cluster_std_list": []
    }

    # Calculate features for each cluster
    for j in range(len(centroids)):
        # Set of traces in X that belong to cluster 'j'
        X_in_cluster = X[y_pred == j]

        try:
            # Calculate radius as maximum distance of a point to centroid
            r["radius_list"].append(distance.cdist(X_in_cluster, [centroids[j]]).max())
        except ValueError:
            r["radius_list"].append(0)

        # Average intra-cluster distances
        dist_intra = distance.pdist(X_in_cluster).mean()
        r["dist_intra_cluster_list"].append(dist_intra)

        # Skewness of cluster
        skewness = skew(X_in_cluster, axis=None)
        r["skewness_list"].append(skewness)

        # Std of cluster
        c_std = X_in_cluster.std()
        r["cluster_std_list"].append(c_std)

    return r


def get_mean_squared_error(clustering_i, clustering_j):
    """
        Calculate features based on the mean squared error between
        clustering_j and clustering_i.

        Parameters:
        -----------
            clustering_i (np.array): Set of centroids at i
            clustering_j (np.array): Set of centroids at the following index
    """
        
    try:
        mse = np.mean((clustering_i - clustering_j) ** 2, axis=0)

        return {
            "total_MSE": np.sum(mse),
            "avg_MSE": np.mean(mse),
            "count_non_zero_MSE": np.count_nonzero(mse)        
        }
    except:
        return {}


def compare_clusterings(resp_1, resp_2):
    """
        Compare two clusterings in consecutive windows and return
        features tracking the evolution of clustering.
        
        Parameters:
        -----------
            resp_1 (dict): Information about clustering at i
            resp_2 (dict): Information about clustering at i + 1
    """
    r = {}

    # If there is no centroid in the two clusterings, return empty
    if len(resp_1["centroids"]) == 0 or len(resp_2["centroids"]) == 0:
        return r

    for key in resp_1:
        try:
            if key != "i":
                key_ = key.replace("_list", "")
            
                # ---------------
                # diff_centroids
                # ---------------
                if key == "centroids":
                    # Calculates the minimum average distance between centroids
                    r["diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).mean(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).mean(),
                    )
                    
                    r["std_diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).std(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).std(),
                    )

                    # Add Mean Squared Error features to the return
                    r.update(
                        get_mean_squared_error(resp_1[key], resp_2[key])
                    )

                # -------------------------
                # diff_radius
                # diff_skewness
                # diff_dist_intra_cluster
                # diff_cluster_std
                # -------------------------
                # For list of individual features per cluster, calculates the 
                # average squared difference between them
                elif isinstance(resp_1[key], list):                    
                    try:
                        r["diff_" + key_] = (
                            (np.array(resp_2[key]) - np.array(resp_1[key])) ** 2
                        ).mean()
                    except ValueError:
                        pass

                else:
                # -----------------
                # diff_DBi
                # diff_Silhouette
                # diff_k
                # -----------------
                # For numeric features, calculate the difference
                    r["diff_" + key_] = resp_2[key] - resp_1[key]
            else:
                r["i"] = resp_2["i"]

        except Exception as e:
            print(key)
            print(resp_1[key])
            raise

    return r

def run_offline_clustering_window(
    model, window, df, sliding_window=False, sliding_step=5
):
    """
        Runs the trace clustering approach based on moving trace windows

        Parameters:
        -----------
                  model (sklearn): Scikit-learn clustering model
                     window (int): Size of the trace window to consider when clustering
                df (pd.DataFrame): Dataset with traces in vector space representation
      sliding_window(bool, False): Whether to use a sliding window or not
            sliding_step(int, 5): Size of the step in the case of sliding window

        Returns:
        --------
            all_metrics (pd.DataFrame): DataFrame with the results of execution, features 
                extracted from trace clustering and resulting centroids
    """
    resp = []

    if sliding_window:
        loop = range(0, len(df) - window + 1, sliding_step)
    else:
        loop = range(0, len(df), window)

    col_names = df.columns

    for i in loop:
        # Selects traces inside the window
        X = df.loc[i : i + window - 1].values
       
        # Fit and predict model to the current window
        y_pred = model.fit_predict(X)
        
        # Centroids
        centers =  (
            pd.DataFrame(X)
            .groupby(y_pred)
            .mean()
            .drop(-1, errors="ignore")
            .values
        )

        # Lookup table to order clusters labels the same way
        #
        idx = np.argsort(centers.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(len(idx))
        
        try:
            y_pred = lut[y_pred]
        except:
            pass
        
        # Recalculate centroids with new ordering
        centers =  (
            pd.DataFrame(X)
            .groupby(y_pred)
            .mean()
            .drop(-1, errors="ignore")
            .values
        )
        
        # Start dictionary to be filled with the results
        r = {"i": i, "k": len(np.unique(y_pred[y_pred > 0]))}

        # Count traces per clusters
        values, counts = np.unique(y_pred, return_counts=True)

        # ----------------------------
        # Calculate validation indexes
        # ----------------------------
        if max(counts) >= 2 and len(values) > 1:
            r.update(get_validation_indexes(X, y_pred))

        # Add centroids to results
        r["centroids"] = centers

        # Inter-clusters distance
        inter_dist = distance.pdist(r["centroids"])
        r["avg_dist_between_centroids"] = inter_dist.mean()
        r["std_dist_between_centroids"] = inter_dist.std()

        # Add features to results
        r["volume_list"] = counts
        r.update(get_centroids_metrics(X, y_pred, r["centroids"]))

        # Add current iteration to full response
        resp.append(r)

    # Turn into dataframe
    run_df = pd.DataFrame(resp).set_index("i")

    # Expand values for individual clusters
    for col in [
        "radius_list",
        "dist_intra_cluster_list",
        "skewness_list",
        "cluster_std_list",
        "volume_list",
    ]:
        min_individuals = run_df[col].apply(len).max()

        try:
            # Create averages
            if col != "volume_list":
                run_df["avg_" + col.replace("_list", "")] = run_df[col].apply(lambda x: np.mean(x))
                run_df["std_" + col.replace("_list", "")] = run_df[col].apply(lambda x: np.std(x))
                
        except Exception as e:
            print(e)
            pass

    # Calculate time-dependent features
    measures = [compare_clusterings(resp[i], resp[i + 1]) for i in range(len(resp) - 1)]
    measures_df = pd.DataFrame(measures).set_index("i")
    measures_df.fillna(0, inplace=True)

    # Merge results
    all_metrics = run_df.join(measures_df)
    all_metrics.index += all_metrics.index[1]

    return all_metrics