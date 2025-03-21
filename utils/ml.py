import numpy as np
from joblib import Memory

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

from external.a2d2.util.TSB_AD.models.norma import NORMA
from external.a2d2.util.util_a2d2 import find_length

import warnings
import pandas as pd


def norm_a(data, flags, areas, z_threshold=-2.5, delta_threshold=0.05, clustering_diff_threshold=0.1):
    """
    Detect mislabels anomaly points using robust z-score, leave-one-out analysis, and clustering.
    """
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    results = []
    mislabel_info = []

    for area in areas:
        start = int(area['start'])
        end = int(area['end'])
        area_data = data[start:end + 1]
        area_flags = np.array(flags[start:end + 1])

        # Compute NormA scores
        area_data_arr = np.array(area_data, dtype=float).flatten()
        pattern_length = find_length(area_data_arr)

        clf_area = NORMA(pattern_length=pattern_length, nm_size=3 * pattern_length, percentage_sel=1,
                         normalize='z-norm')
        clf_area.fit(area_data_arr)
        area_scores = np.array(clf_area.decision_scores_)
        area_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(area_scores.reshape(-1, 1)).ravel()
        pad = (pattern_length - 1) // 2
        area_scores = np.array([area_scores[0]] * pad + list(area_scores) + [area_scores[-1]] * pad)

        # Initial threshold
        if np.any(area_flags == 1):
            original_threshold = np.min(area_scores[area_flags == 1])
        else:
            original_threshold = np.percentile(area_scores, 95)

        anomaly_mask = area_scores >= original_threshold
        original_anomaly_count = np.sum(anomaly_mask)

        # Robust z-score + leave-one-out
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
                if len(remaining) > 0:
                    loo_threshold = np.min(remaining)
                else:
                    loo_threshold = original_threshold
                delta = loo_threshold - original_threshold
                if z < z_threshold or delta > delta_threshold:
                    sensitivity_mislabels.append(int(idx))

        # KMeans clustering
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

        # Merge mislabels and sort by score
        combined_mislabels = list(set(sensitivity_mislabels).union(set(clustering_mislabels)))
        combined_mislabels.sort()
        mislabel_scores = [(int(start + idx), float(area_scores[idx])) for idx in combined_mislabels]
        mislabel_scores.sort(key=lambda x: x[1])  # sort by score low to high

        # Convert to global indices
        global_mislabels = [int(start + idx) for idx in combined_mislabels]

        # Recompute threshold after removing mislabels
        new_area_flags = area_flags.copy()
        for idx in combined_mislabels:
            new_area_flags[idx] = 0
        if np.any(new_area_flags == 1):
            new_threshold = np.min(area_scores[new_area_flags == 1])
        else:
            new_threshold = np.percentile(area_scores, 95)

        new_anomaly_mask = area_scores >= new_threshold
        new_anomaly_count = np.sum(new_anomaly_mask)
        reduction_count = original_anomaly_count - new_anomaly_count
        reduction_ratio = (reduction_count / original_anomaly_count) if original_anomaly_count > 0 else 0

        # Update flags for system-detected anomalies
        for i, is_anom in enumerate(new_anomaly_mask):
            if is_anom and flags[start + i] != 1:
                flags[start + i] = 3

        # Save result
        results.append({
            'start': int(start),
            'end': int(end),
            'original_threshold': float(original_threshold),
            'new_threshold': float(new_threshold),
            'original_anomaly_count': int(original_anomaly_count),
            'new_anomaly_count': int(new_anomaly_count),
            'reduction_count': int(reduction_count),
            'reduction_ratio': float(reduction_ratio),
        })
        mislabel_info.append({
            'area_start': int(start),
            'area_end': int(end),
            'potential_mislabels_global_indices': [int(x) for x, _ in mislabel_scores],
            'potential_mislabels_scores_sorted': mislabel_scores
        })

    print(results, mislabel_info)

    return {'results': results, 'mislabel_info': mislabel_info, 'flags': flags.tolist()}

