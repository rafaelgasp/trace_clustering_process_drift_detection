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
        Retorna indíces de validação de clustering baseados
        nos inputs X e os grupos associados a cada cado y_pred

        Silhouette, DBScan
    """
    return {
        "Silhouette": silhouette_score(X, y_pred),
        "DBi": davies_bouldin_score(X, y_pred),
        #  "DBCV": DBCV(X, y_pred)
    }


def get_centroids_metrics(X, y_pred, centroids):
    """
        Calcula métricas baseadas nos centróides e nos grupos 
        atribuídos a cada cado no conjunto X

        Parâmetros:
        -----------
            X (pd.DataFrame): Conjunto de dados 
            y_pred (np.array): Grupos atribuídos a cada cluster
            centroids (np.array): Centróides de cada grupo
        
        Retorno:
        ---------
            dict
    """
    r = {
        "radius_list": [],
        "dist_intra_cluster_list": [],
        "skewness_list": [],
        "cluster_std_list": [],
        # "volume_list": {values[i]: counts[i] for i in range(len(values))},
    }

    # Calculates metrics for each cluster
    for j in range(len(centroids)):
        X_in_cluster = X[y_pred == j]

        try:
            # Maximum distance of a point to centroid
            r["radius_list"].append(distance.cdist(X_in_cluster, [centroids[j]]).max())
        except ValueError:
            r["radius_list"].append(0)

        # Average intra-cluster distances
        dist_intra = distance.pdist(X_in_cluster).mean()
        r["dist_intra_cluster_list"].append(dist_intra)
        # r["dist_intra_cluster_i=" + str(j)] = dist_intra

        # Skewness of cluster
        skewness = skew(X_in_cluster, axis=None)
        r["skewness_list"].append(skewness)
        # r["skewness_i=" + str(j)] = skewness

        # Std of cluster
        c_std = X_in_cluster.std()
        r["cluster_std_list"].append(c_std)
        # r["cluster_std_i=" + str(j)] = c_std
    return r


def get_mean_squared_error(centroids_1, centroids_2, cols=None):
    if cols is None:
        cols = list(range(centroids_1.shape[1]))
        
    try:
        mse = np.mean((centroids_1 - centroids_2) ** 2, axis=0)
        # sse = np.sum((centroids_1 - centroids_2) ** 2, axis=0)

        #resp = {
        #    "MSE__" + str(cols[i]): mse[i] for i in range(len(cols))
        #}
        # resp.update({
        #     "SSE__" + str(cols[i]): sse[i] for i in range(len(cols))
        # })    

        # centroids_mse = np.mean((centroids_1 - centroids_2) ** 2, axis=1)
        # centroids_sse = np.sum((centroids_1 - centroids_2) ** 2, axis=1)
        # for i in range(len(centroids_mse)):
        #    resp["centroid_" + str(i) + "_MSE"] = centroids_mse[i]
        #    resp["centroid_" + str(i) + "_SSE"] = centroids_sse[i]

        resp = {
            "total_MSE": np.sum(mse),
            "avg_MSE": np.mean(mse),
            # "total_SSE": np.sum(sse),
            # "avg_SSE": np.mean(sse),
            "count_non_zero_MSE": np.count_nonzero(mse)        
        }
    except:
        return {}
    return resp


def compare_clusterings(resp_1, resp_2, cols=None):
    """
        Compara dois agrupamentos em duas janelas diferentes e
        retorna métricas das variações 
    """
    r = {}

    # se não houve centroides encontrados nos dois grupos, retorna vazio
    if len(resp_1["centroids"]) == 0 or len(resp_2["centroids"]) == 0:
        return r

    for key in resp_1:
        try:
            if key != "i":
                key_ = key.replace("_list", "")
            
                # ---------------
                # diff_centroids
                # ---------------
                # Calcula a menor distância média interclusters os clusters
                if key == "centroids":
                    r["diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).mean(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).mean(),
                    )
                    
                    r["std_diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).std(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).std(),
                    )

                    r.update(
                        get_mean_squared_error(resp_1[key], resp_2[key], cols)
                    )

                if key == "linear_sum" or key == "squared_sum":
                    r["diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).mean(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).mean(),
                    )
                    
                    r["std_diff_" + key_] = min(
                        distance.cdist(resp_1[key], resp_2[key]).min(axis=0).std(),
                        distance.cdist(resp_2[key], resp_1[key]).min(axis=0).std(),
                    )

                # -------------------------
                # diff_radius
                # diff_skewness
                # diff_dist_intra_cluster
                # diff_cluster_std
                # -------------------------
                # Para métricas baseadas em lista, calcula a diferença média
                elif isinstance(resp_1[key], list):
                    # Reordena os valores do cluster_1 para correspondência
                    # resp_1[key] = np.array(resp_1[key][ordem]).tolist()
                    
                    try:
                        r["diff_" + key_] = (
                            (np.array(resp_2[key]) - np.array(resp_1[key])) ** 2
                        ).mean()
                    except ValueError:
                        pass

                    # Metricas individuais por cluster
                    # r.update(get_individuals(resp_1, resp_2, key))


                # ------------
                # diff_volume
                # ------------
                # Para o volume, é necessário separar caso haja o grupo -1
                # (outliers no DBScan)
                # De outra forma, pega a menor distância média
                # elif key == "volume_list":                    
                #     if -1 in resp_1[key].keys():
                #         r["diff_volume_outliers"] = (
                #             resp_2[key][-1] - resp_1[key][-1]
                #         ) / sum(resp_2[key].values())

                #     r["diff_volume"] = (
                #         distance.cdist(
                #             [[x] for x in resp_1[key].values()],
                #             [[x] for x in resp_2[key].values()],
                #         )
                #         .min(axis=0)
                #         .mean()
                #     )

                #     # Metricas individuais por cluster
                #     # r.update(get_individuals(resp_1, resp_2, key))
                else:
                # -----------------
                # diff_DBi
                # diff_Silhouette
                # diff_k
                # -----------------
                # Para métricas númericas, faz diretamente a subtração
                    r["diff_" + key_] = resp_2[key] - resp_1[key]
            else:
                # --
                # i
                # --
                # Para o i, segue o valor do segundo momento
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
        Roda o modelo de clusterização offline baseado em janelas

        Parâmetros:
        -----------
                  model (sklearn): Modelo parametrizado
                     window (int): Tamanho da janela a ser utilizada
                df (pd.DataFrame): Base de dados a ser utilizada 
      sliding_window(bool, False): Se utiliza janela deslizante ou não
            sliding_step(int, 50): Tamanho do passo na janela deslizante

        Retorno:
        --------
            run_df, measures_df
    """
    resp = []
    # old_X = None

    if sliding_window:
        # loop = tqdm_notebook(range(0, len(df) - window + 1, sliding_step))
        loop = range(0, len(df) - window + 1, sliding_step)
    else:
        # loop = tqdm_notebook(range(0, len(df), window))
        loop = range(0, len(df), window)

    # prev_centers = None
    for i in loop:
        # print(i)
        # Seleciona janela olhando para frente
        X = df.loc[i : i + window - 1].values
        # print(i, X.shape)

        # Predita modelo com a normalização dos números dos clusters
        # model.fit(X)
        # y_pred = model.labels_

        # if reuse_centers and prev_centers is not None:
        #     model.set_params(**{
        #         'init': prev_centers,
        #         'n_init': 1
        #     })

        y_pred = model.fit_predict(X)
        
        centers =  (
            pd.DataFrame(X)
            .groupby(y_pred)
            .mean()
            .drop(-1, errors="ignore")
            .values
        )
        # Faz uma lookup table para reorganizar a ordem das labels
        # dos clusters
        idx = np.argsort(centers.sum(axis=1))
        lut = np.zeros_like(idx)
        lut[idx] = np.arange(len(idx))
        
        try:
            y_pred = lut[y_pred]
        except:
            pass
        
        centers =  (
            pd.DataFrame(X)
            .groupby(y_pred)
            .mean()
            .drop(-1, errors="ignore")
            .values
        )
        
        #print(i)
        #print("Y_PRED", y_pred)
        #print("NEW_Y_PRED", y_pred)

        # Monta dicionário com respostas dos métodos
        r = {"i": i, "k": len(np.unique(y_pred[y_pred > 0]))}

        # Contagem de dados por labels preditas
        values, counts = np.unique(y_pred, return_counts=True)

        # Adaptative Window
        # if len(ref_runs) > 0:
        #    t1 = len(pd.concat([ref_runs, det_runs]).unique())
        #    t2 = len(pd.concat([new_ref_runs, new_det_runs]).unique())
        #    
        #    var_ratio = t2 / t1
        #    
        #    old_window_size = current_window_size
        #    current_window_size = int(current_window_size * var_ratio)

        # ----------------------------------
        # Calcula métricas silhouette e DBi
        # ----------------------------------
        if max(counts) >= 2 and len(values) > 1:
            r.update(get_validation_indexes(X, y_pred))

        r["centroids"] = centers
        # r.update(get_individual_dimensions(centers, cols=df.columns))

        # if old_centroids is not None:
        #    ordem = distance.cdist(old_centroids, r["centroids"]).argmin(axis=1)
        #    r["centroids"] = r["centroids"][ordem]

        inter_dist = distance.pdist(r["centroids"])
        # r["sum_dist_between_centroids"] = inter_dist.sum()
        r["avg_dist_between_centroids"] = inter_dist.mean()
        r["std_dist_between_centroids"] = inter_dist.std()
        # r["min_dist_between_centroids"] = inter_dist.min()
        # r["max_dist_between_centroids"] = inter_dist.max()
        # r["amplitude_dist_between_centroids"] = r["max_dist_between_centroids"] - r["min_dist_between_centroids"]
        

        #r["linear_sum"] = pd.DataFrame(X).groupby(y_pred).sum().values
        #r["sq_sum_linear_sum"] = (r["linear_sum"] ** 2).sum()

        # r["squared_sum"] = (pd.DataFrame(X) ** 2).groupby(y_pred).sum().values
        # r["sum_squared_sum"] =  r["squared_sum"].sum()  
        # r["squared_sum"].mean(axis=1)

        # inter_dist = distance.pdist(r["squared_sum"])
        # r["avg_dist_between_sq_sums"] = inter_dist.mean()
        # r["std_dist_between_sq_sums"] = inter_dist.std()

        r["volume_list"] = counts

        #for i in range(len(centers)):
        #    for j in range(i + 1, len(centers)):
        #        r["dist_between_" + str(i) + "_" + str(j)] = distance.euclidean(centers[i], centers[j])
        
        r.update(get_centroids_metrics(X, y_pred, r["centroids"]))

        # Adiciona iteração atual na resposta
        resp.append(r)

        # prev_centers = centers

    run_df = pd.DataFrame(resp).set_index("i")

    # Expand values for individual clusters
    for col in [
        "radius_list",
        "dist_intra_cluster_list",
        "skewness_list",
        "cluster_std_list",
       # "squared_sum",
       # "linear_sum"
        "volume_list",
    ]:
        min_individuals = run_df[col].apply(len).max()

        try:
            # if col in ["squared_sum"]: #, "linear_sum"]:
            #     for i in range(min_individuals):
            #         run_df[col.replace("_list", "") + "_i=" + str(i)] = run_df[col].apply(
            #             lambda x: x[i] if i < len(x) else np.nan
            #         )

            # Create averages
            if col != "volume_list":
                run_df["avg_" + col.replace("_list", "")] = run_df[col].apply(lambda x: np.mean(x))
                run_df["std_" + col.replace("_list", "")] = run_df[col].apply(lambda x: np.std(x))
                
        except Exception as e:
            print(e)
            pass

    measures = [compare_clusterings(resp[i], resp[i + 1], df.columns) for i in range(len(resp) - 1)]
    measures_df = pd.DataFrame(measures).set_index("i")
    measures_df.fillna(0, inplace=True)

    all_metrics = run_df.join(measures_df)
    all_metrics.index += all_metrics.index[1]

    return all_metrics