import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats


def get_metrics(drifts, resp, window_size=100, verbose=False):
    """
    Precision, Recall, F1 Score and Delay
    for the drift predictions
    """
    precision = 0
    recall = 0
    tp = 0
    delay = 0
    avg_delay = 0
    resp_ = resp.copy()
    predicted = [0 for x in resp_]
    
    if isinstance(window_size, int):
        window_size = np.repeat(window_size, len(drifts))
    
    for i in range(len(drifts)):    
        for j in range(len(resp_)):
            # print(drifts[i], resp_[j], drifts[i] - resp_[j])
            
            if 0 <= drifts[i] - resp_[j] <= 2 * window_size[i]:
                if verbose:
                    print((drifts[i], drifts[i] + window_size[i], resp_[j]))
                    
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


def exp_mw_avg_std(value, a):
    mrt_array = np.array(value)
    M = len(mrt_array)
    weights = (1-a)**np.arange(M-1, -1, -1) # This is reverse order to match Series order
    ewma = sum(weights * mrt_array) / sum(weights)
    bias = sum(weights)**2 / (sum(weights)**2 - sum(weights**2))
    ewmvar = bias * sum(weights * (mrt_array - ewma)**2) / sum(weights)
    ewmstd = np.sqrt(ewmvar)
    return ewma, ewmstd


def detect_concept_drift(
    df, var_ref, rolling_window=5, std_tolerance=3, min_tol=0.0025, verbose=False
):
    window_buffer = []
    drifts = []
    mean = None
    std = None

    lowers = []
    uppers = []
    if use_median:
        means = df[var_ref].rolling(window=rolling_window).median().values.tolist()
    else:    
        means = df[var_ref].rolling(window=rolling_window).mean().values.tolist()

    for i, row in df.iterrows():
        if len(window_buffer) < rolling_window:
            window_buffer.append(row[var_ref])
            lowers.append(np.nan)
            uppers.append(np.nan)

        else:
            if mean is not None:
                if mean == 0:
                    mean == 1

                expected_lower = min(
                    mean - (std_tolerance * std),
                    (1 - min_tol) * mean
                )
                expected_upper = max(
                    mean + (std_tolerance * std),
                    (1 + min_tol) * mean
                )

                lowers.append(expected_lower)
                uppers.append(expected_upper)

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