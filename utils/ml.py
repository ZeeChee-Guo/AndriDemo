import numpy as np
from joblib import Memory

from sklearn.preprocessing import MinMaxScaler
import warnings
import pandas as pd

from external.a2d2.util.TSB_AD.models.norma import NORMA
from external.a2d2.util.util_a2d2 import find_length

import io
import contextlib


memory = Memory(location="./norma_cache", verbose=0)



@memory.cache
def compute_global_scores(data_arr, pattern_length=None):
    """Fit NORMA and return pattern_length and normalized scores with padding."""

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        if pattern_length is None:
            pattern_length = find_length(data_arr)

        clf_global = NORMA(
            pattern_length=pattern_length,
            nm_size=3 * pattern_length,
            percentage_sel=1,
            normalize='z-norm'
        )
        clf_global.fit(data_arr)
        global_scores = np.array(clf_global.decision_scores_)
        global_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(global_scores.reshape(-1, 1)).ravel()
        pad = (pattern_length - 1) // 2
        padded_scores = np.array([global_scores[0]] * pad + list(global_scores) + [global_scores[-1]] * pad)
    return pattern_length, padded_scores



def norm_a_scoring(data, flags, areas, pattern_length=None):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1, 3, 4])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores = compute_global_scores(data_arr, pattern_length=pattern_length)


    all_area_scores = []
    for area in areas:
        s, e = int(area['start']), int(area['end'])
        all_area_scores.append(global_scores[s:e+1])
    all_area_scores = np.concatenate(all_area_scores) if all_area_scores else np.array([])

    area_75_percentile = float(np.percentile(all_area_scores, 75))

    user_flagged_idxs = np.where(flags == 1)[0]
    user_flagged_scores = global_scores[user_flagged_idxs]
    user_25_percentile = float(np.percentile(user_flagged_scores, 25))

    threshold = (area_75_percentile + user_25_percentile) / 2

    # Create a mask of indices that are in areas
    area_mask = np.zeros_like(flags, dtype=bool)
    for area in areas:
        start, end = int(area['start']), int(area['end'])
        area_mask[start:end+1] = True

    # Reset all flags outside areas to 0
    flags[~area_mask] = 0

    # Update flags within areas based on threshold
    for i in np.where(area_mask)[0]:
        score = global_scores[i]
        if flags[i] == 1 and score < threshold:
            flags[i] = 3
        elif flags[i] in [0, 3] and score >= threshold:
            flags[i] = 4


    return {
        'global_scores': list(global_scores),
        'pattern_length': int(pattern_length),
        'training_set_75_percentile': float(area_75_percentile),
        'user_25_percentile': float(user_25_percentile),
        'threshold': float(threshold),
        'flags': list(map(int, flags)),
    }