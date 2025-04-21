import numpy as np
from joblib import Memory

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings
import pandas as pd

from external.a2d2.util.TSB_AD.models.norma import NORMA
from external.a2d2.util.util_a2d2 import find_length

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



def norm_a_scoring(data, flags, areas, z_threshold=-2.5, delta_threshold=0.07,
                   fixed_n=5, prop=0.5, cohens_d_threshold=0.1):


    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1, 3, 4])] = 1

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
        if flagged_indices.size > 0:
            fscores = area_scores[flagged_indices]
            median_score = np.median(fscores)
            mad = np.median(np.abs(fscores - median_score)) or 1e-6
            for i, idx in enumerate(flagged_indices):
                z = (area_scores[idx] - median_score) / mad
                loo_thr = np.min(np.delete(fscores, i)) if fscores.size > 1 else original_threshold
                if z < z_threshold or (loo_thr - original_threshold) > delta_threshold:
                    sensitivity_mislabels.append(int(idx))

        window_mislabels = []
        segments = []
        if flagged_indices.size > 0:
            run_start = flagged_indices[0]
            prev = run_start
            for idx in flagged_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    segments.append((run_start, prev))
                    run_start = idx
                    prev = idx
            segments.append((run_start, prev))

        for seg_start, seg_end in segments:
            L = seg_end - seg_start + 1
            win_len = max(fixed_n, int(np.ceil(prop * L)))
            win_start = max(0, seg_start - win_len)
            win_end = min(len(area_scores) - 1, seg_end + win_len)

            seg_scores = area_scores[seg_start:seg_end+1]
            ext_idxs = np.concatenate([
                np.arange(win_start, seg_start),
                np.arange(seg_end+1, win_end+1)
            ])
            ext_scores = area_scores[ext_idxs] if ext_idxs.size > 0 else np.array([])

            mean_seg, std_seg = np.mean(seg_scores), np.std(seg_scores, ddof=1)
            if ext_scores.size > 0:
                mean_ext, std_ext = np.mean(ext_scores), np.std(ext_scores, ddof=1)
            else:
                mean_ext, std_ext = mean_seg, std_seg

            dof = seg_scores.size + ext_scores.size - 2
            pooled_var = ((seg_scores.size - 1) * std_seg**2 +
                          (ext_scores.size - 1) * std_ext**2) / dof if dof > 0 else std_seg**2
            pooled_sd = np.sqrt(pooled_var) if pooled_var > 0 else 1e-6
            cohens_d = (mean_seg - mean_ext) / pooled_sd

            if abs(cohens_d) < cohens_d_threshold:
                window_mislabels.extend(range(seg_start, seg_end+1))

        combined = sorted(set(sensitivity_mislabels) | set(window_mislabels))
        union_mask = (area_flags == 1) | (np.isin(np.arange(len(area_flags)), combined))
        threshold = np.min(area_scores[union_mask]) if union_mask.any() else np.percentile(area_scores, 95)

        for idx in combined:
            flags[start + idx] = 3
        for i, score in enumerate(area_scores):
            if score >= threshold and flags[start + i] not in [1, 3]:
                flags[start + i] = 4

    flags_list = flags.tolist()
    scores_list = list(global_scores)

    return {
        'flags': flags_list,
        'global_scores': scores_list,
    }

