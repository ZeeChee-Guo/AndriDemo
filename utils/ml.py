import numpy as np
from joblib import Memory

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from external.a2d2.util.TSB_AD.models.norma import NORMA
from external.a2d2.util.util_a2d2 import find_length

import warnings
import pandas as pd
import io
import contextlib


memory = Memory(location="./norma_cache", verbose=0)



@memory.cache
def compute_global_scores(data_arr):
    """Fit NORMA and return pattern_length and normalized scores with padding."""

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
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



def norm_a_scoring(data, flags, areas, z_threshold=-2.5, delta_threshold=0.07, clustering_diff_threshold=0.1):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1, 3, 4])] = 1
    suspect_indices = []

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores = compute_global_scores(data_arr)



    for area in areas:
        start = int(area['start'])
        end = int(area['end'])

        area_flags = flags[start:end+1].copy()
        area_scores = global_scores[start:end+1]

        if np.any(area_flags == 1):
            original_threshold = np.min(area_scores[area_flags == 1])
        else:
            original_threshold = np.percentile(area_scores, 95)

        flagged_indices = np.where(area_flags == 1)[0]
        sensitivity_mislabels = []
        if len(flagged_indices) > 0:
            flagged_scores = area_scores[flagged_indices]
            median_score = np.median(flagged_scores)
            mad = np.median(np.abs(flagged_scores - median_score))
            if mad == 0:
                mad = 1e-6
            for i, idx in enumerate(flagged_indices):
                score_i = area_scores[idx]
                z = (score_i - median_score) / mad
                remaining = np.delete(flagged_scores, i)
                loo_threshold = np.min(remaining) if len(remaining) > 0 else original_threshold
                delta = loo_threshold - original_threshold
                if z < z_threshold or delta > delta_threshold:
                    sensitivity_mislabels.append(int(idx))

        clustering_mislabels = []
        if len(flagged_indices) > 1:
            flagged_scores = area_scores[flagged_indices]
            X = flagged_scores.reshape(-1, 1)
            kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_.ravel()
            lower_cluster = int(np.argmin(centers))
            if np.abs(centers[0] - centers[1]) > clustering_diff_threshold:
                clustering_mislabels = [int(idx) for idx in flagged_indices[labels == lower_cluster]]

        combined_mislabels = sorted(set(sensitivity_mislabels).union(set(clustering_mislabels)))

        for idx in combined_mislabels:
            suspect_indices.append(start + idx)

        union_mask = (area_flags == 1) | (np.isin(np.arange(len(area_flags)), combined_mislabels))
        threshold = np.min(area_scores[union_mask]) if np.any(union_mask) else np.percentile(area_scores, 95)

        for idx in combined_mislabels:
            flags[start + idx] = 3

        for i in range(len(area_scores)):
            if area_scores[i] >= threshold and flags[start + i] not in [1, 3]:
                flags[start + i] = 4

    suspect_indices = sorted(set(suspect_indices))

    score_98 = float(np.percentile(global_scores, 98))
    score_92 = float(np.percentile(global_scores, 84))
    score_85 = float(np.percentile(global_scores, 70))

    return {
        'suspects': suspect_indices,
        'flags': flags.tolist(),
        'global_scores': global_scores.tolist(),
        'score_98': score_98,
        'score_92': score_92,
        'score_85': score_85
    }


