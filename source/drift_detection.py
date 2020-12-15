import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats


def get_metrics(drifts: list, resp: list, window_size=100, verbose=False) -> dict:
    """
    Given the drifts predicted and the ground truth, calculates binary classification metrics 
    (Precision, Recall, F1 Score) and delay of detection. We consider a drift to be detected
    correctly when it has been found within 2 * window_size after its true index. 

    Parameters:
    ------------
        drifts (list): List with the index of the detected drifts
        resp (list): List with the index of the ground truth drifts
        window_size (int): Size of the trace clustering window used to consider the delay.
        verbose (bool): Whether to print results

    Returns:
    ---------
        dict: dictionary with the value of each metric
    """
    # Initialize metrics with 0
    precision = 0
    recall = 0
    tp = 0
    delay = 0
    avg_delay = 0
    resp_ = resp.copy()
    predicted = [0 for x in resp_]
    
    # Transforms the window_size into a vector with the size of the drifts found
    if isinstance(window_size, int):
        window_size = np.repeat(window_size, len(drifts))
    
    # Iterates over all drifts found and to all ground truths drifts
    for i in range(len(drifts)):    
        for j in range(len(resp_)):            
            # check if the drift found is within 2 * window_size after its true index
            if 0 <= drifts[i] - resp_[j] <= 2 * window_size[i]:
                if verbose:
                    print((drifts[i], drifts[i] + window_size[i], resp_[j]))
                
                # drift found correctly
                delay += drifts[i] - resp_[j]
                tp += 1
                resp_[j] = np.inf
                predicted[j] = 1
                break
    
    if len(drifts) > 0:
        precision = tp/len(drifts)

    if len(resp_) > 0:
        recall = tp/len(resp_)
        
    try:
        f1 = scipy.stats.hmean([precision, recall])
    except ValueError:
        f1 = 0.0
        
    if tp > 0:
        avg_delay = delay/tp
    
    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Delay": avg_delay,
        "Correct_Predictions": predicted,
        "Support": sum(predicted),
        "Drifts_Found": drifts,
        "Resp": resp,
    }


def detect_concept_drift(
    df, var_ref, rolling_window=5, std_tolerance=3, min_tol=0.0025, verbose=False
):
    """
        Performs the drift detection algorithm. Based on the features measured from 
        tracking the evolution of the trace clustering over the windows, estimate 
        tolerance boundaries to detect a drift when the feature value measured at the
        current index lies outside of the boundaries. 

        Parameters:
        ------------
            df (pd.DataFrame): DataFrame of features from tracking the trace clusterings
            var_ref (str): Name of the feature to apply the algorithm. It has to be present
                in the df.columns
            rolling_window (int): The number of rolling windows to consider when estimating
                the tolerance boundaries and detecting the drifts. It smooths the analyzed 
                feature to reduce false positive detections due to noise. 
            std_tolerance (int): Number of times the rolling standard deviation used to 
                the tolerance boundaries. A higher value provides higher tolerance and 
                lower sensitivity to drifts.
            min_tol (float [0.0 - 1.0]): Number of times the rolling average to calculate a
                minimum tolerance boundaries. Useful for not detecting false positives in more 
                stable feature values when the tolerance could be too little.
            verbose (bool): Whether to print the index of drifts as they are found
        
        Returns:
        ---------
            list: List of index of the detected drifts
            dict: Dictionary with the rolling average, lowers and uppers boundaries 
                to assist plotting
    """
    # Initialize variables
    window_buffer = []
    drifts = []
    mean = None
    std = None

    # Lists to keep the rolling average and the lower and upper boundaries to 
    # be returned at the end of the execution of this method to support the plots
    lowers = []
    uppers = []
    means = df[var_ref].rolling(window=rolling_window).mean().values.tolist()

    # Iterates over the values
    for i, row in df.iterrows():
        # If the rolling window is of the desired size
        if len(window_buffer) < rolling_window:
            window_buffer.append(row[var_ref])
            lowers.append(np.nan)
            uppers.append(np.nan)

        else:
            if mean is not None:
                # To avoid errors in multiplication with 0
                if mean == 0:
                    mean == 1

                # Calculates tolerance boundaries considering the rolling mean and std
                expected_lower = min(
                    mean - (std_tolerance * std),
                    (1 - min_tol) * mean
                )
                expected_upper = max(
                    mean + (std_tolerance * std),
                    (1 + min_tol) * mean
                )

                # Adds into the list to return in the end
                lowers.append(expected_lower)
                uppers.append(expected_upper)

                # Checks whether the current value lies outside the tolerance boundaries
                if expected_lower > row[var_ref] or row[var_ref] > expected_upper:
                    if verbose:
                        print(i, expected_lower, expected_upper, row[var_ref])
                    drifts.append(i)

                    window_buffer = []
            else:
                lowers.append(np.nan)
                uppers.append(np.nan)

            if len(window_buffer) > 0:
                window_buffer.pop(0)

            window_buffer.append(row[var_ref])
            if i in drifts:
                mean = None
                std = None
            else:
                mean = np.mean(window_buffer)
                std = np.std(window_buffer)

    return drifts, {"lowers": lowers, "uppers": uppers, "means": means}