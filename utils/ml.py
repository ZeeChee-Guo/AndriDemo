import numpy as np
from joblib import Memory
import diptest

from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture
import warnings
import pandas as pd
from external.a2d2.util.TSB_AD.models.norma import NORMA
from external.a2d2.util.util_a2d2 import find_length
from TSB_UAD.models.sand import SAND
from TSB_UAD.utils.slidingWindows import find_length as find_sand_length

memory = Memory(location="./norma_cache", verbose=0)


@memory.cache
def compute_global_scores(data_arr, pattern_length=None):
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

    return pattern_length, padded_scores, clf_global.normalmodel[0], clf_global.normalmodel[1]


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x['start'])
    merged = [intervals[0]]
    for curr in intervals[1:]:
        prev = merged[-1]
        if curr['start'] <= prev['end']:
            prev['end'] = max(prev['end'], curr['end'])
        else:
            merged.append(curr)
    return merged


@memory.cache
def compute_training_set_sand_scores(data_arr, training_set, slidingWindow=None):
    if slidingWindow is None:
        slidingWindow = find_sand_length(data_arr)
    merged_intervals = merge_intervals(training_set)
    train_indices = []
    for interval in merged_intervals:
        train_indices.extend(range(interval['start'], interval['end'] + 1))
    train_indices = np.array(sorted(set(train_indices)), dtype=int)
    train_data = data_arr[train_indices]

    clf = SAND(
        pattern_length=slidingWindow,
        subsequence_length=4 * slidingWindow,
    )
    clf.fit(train_data, overlaping_rate=int(1.5 * slidingWindow))
    scores = clf.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(scores.reshape(-1, 1)).ravel()
    scores_full = np.zeros(len(data_arr))
    scores_full[train_indices] = scores

    print(len(scores_full))
    return slidingWindow, scores_full, [c[0] for c in clf.clusters], clf.weights


def compute_training_set_damp_scores():
    pass


def sand_scoring(data, flags, training_set, pattern_length=None):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1, 3])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores, nms, weights = compute_training_set_sand_scores(
        data_arr, training_set, pattern_length
    )

    nms = [np.array(nm).tolist() for nm in nms]
    weights = list(np.array(weights).flatten())


    return {
        'global_scores': list(global_scores),
        'pattern_length': int(pattern_length),
        'flags': list(map(int, flags)),
        'nms': nms,
        'weights': weights,
    }


def norm_a_scoring(data, flags, pattern_length=None):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1, 3])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores, nms, weights = compute_global_scores(data_arr, pattern_length=pattern_length)

    return {
        'global_scores': list(global_scores),
        'pattern_length': int(pattern_length),
        'flags': list(map(int, flags)),
        'nms': nms,
        'weights': weights,
    }


# def damp_scoring(data, flags, pattern_length=None):
#     warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
#     flags = np.array(flags)
#     flags[np.isin(flags, [1, 3])] = 1
#
#     data_arr = np.array(data, dtype=float).flatten()
#     pattern_length, global_scores, nms, weights = compute_global_scores(data_arr, pattern_length=pattern_length)
#
#     # miu_list, sigma_list = fit_user_labels(global_scores, flags)
#
#     return {
#         'global_scores': list(global_scores),
#         'pattern_length': int(pattern_length),
#         'flags': list(map(int, flags)),
#         'nms': nms,
#         'weights': weights,
#     }


def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    eps = 1e-8
    sigma1 = max(sigma1, eps)
    sigma2 = max(sigma2, eps)
    return np.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2 * sigma2 ** 2) - 0.5


def cohen_d(mu1, sigma1, mu2, sigma2):
    pooled_std = ((sigma1 ** 2 + sigma2 ** 2) / 2) ** 0.5
    if pooled_std == 0:
        return float('inf')
    return abs(mu1 - mu2) / pooled_std


def find_gaussian_intersection(mu1, sigma1, mu2, sigma2):
    if mu1 > mu2:
        mu1, mu2 = mu2, mu1
        sigma1, sigma2 = sigma2, sigma1

    a = 1 / (2 * sigma1 ** 2) - 1 / (2 * sigma2 ** 2)
    b = mu2 / (sigma2 ** 2) - mu1 / (sigma1 ** 2)
    c = (mu1 ** 2) / (2 * sigma1 ** 2) - (mu2 ** 2) / (2 * sigma2 ** 2) - np.log(sigma2 / sigma1)
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            x = -c / b
            if mu1 < x < mu2:
                return x
            else:
                return (mu1 + mu2) / 2
        else:
            return (mu1 + mu2) / 2
    delta = b ** 2 - 4 * a * c
    if delta < 0:
        return (mu1 + mu2) / 2
    sqrt_delta = np.sqrt(delta)
    x1 = (-b + sqrt_delta) / (2 * a)
    x2 = (-b - sqrt_delta) / (2 * a)
    candidates = [x for x in [x1, x2] if mu1 < x < mu2]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 2:
        return min(candidates)
    else:
        return (mu1 + mu2) / 2


def fit_user_labels(scores, flags, training_set):
    scores = np.asarray(scores)

    masks = (flags == 1)
    scores_sel = scores[masks].reshape(-1, 1)
    bgmm = BayesianGaussianMixture(
        n_components=5,
        weight_concentration_prior_type='dirichlet_process',
        random_state=0
    )
    bgmm.fit(scores_sel)

    weight_threshold = 0.2
    weights = bgmm.weights_
    means = bgmm.means_.flatten()
    covariances = bgmm.covariances_.flatten()
    stds = np.sqrt(covariances)

    active_idx = np.where(weights > weight_threshold)[0]
    if len(active_idx) < 2:
        print('剩下不到两个活跃成分:')
        for idx in active_idx:
            print(f"  mean: {means[idx]:.4f}, std: {stds[idx]:.4f}, weight: {weights[idx]:.4f}")
        return 1, [-1]

    labels = bgmm.predict(scores_sel)
    mask_active_points = np.isin(labels, active_idx)
    scores_active = scores_sel[mask_active_points].flatten()
    if scores_active.size >= 2:
        scores_active_sorted = np.sort(scores_active)
        dip_stat, pval = diptest.diptest(scores_active_sorted)
        print(f"[只取活跃成分标记点] Dip Statistic: {dip_stat:.4e}, P-Value: {pval}")
        num_flagged = np.sum(flags == 1)
        print(num_flagged)
        if pval >= 0.05 and num_flagged >= 150:
            print("Diptest -> p >= 0.05，视作单峰合并")
            return 1, [0]
    else:
        print("活跃成分内的已标记点不足，跳过 Dip 测试")

    for idx in active_idx:
        print(f"  mean: {means[idx]:.4f}, std: {stds[idx]:.4f}, weight: {weights[idx]:.4f}")

    active_sorted = sorted(active_idx, key=lambda idx: means[idx])

    MAX_ONCE = 20

    def compute_intersection(idx1, idx2):
        mu1, sigma1, w1 = means[idx1], stds[idx1], weights[idx1]
        mu2, sigma2, w2 = means[idx2], stds[idx2], weights[idx2]

        a = 1.0 / (sigma2 ** 2) - 1.0 / (sigma1 ** 2)
        b = (-2 * mu2) / (sigma2 ** 2) + (2 * mu1) / (sigma1 ** 2)
        C = 2 * (np.log(w2 / sigma2) - np.log(w1 / sigma1))
        c = (mu2 ** 2) / (sigma2 ** 2) - (mu1 ** 2) / (sigma1 ** 2) - C

        if abs(a) < 1e-12:
            if abs(b) < 1e-12:
                return (mu1 + mu2) / 2.0
            return -c / b
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            return (mu1 + mu2) / 2.0
        elif abs(delta) < 1e-12:
            return -b / (2 * a)
        else:
            root1 = (-b + np.sqrt(delta)) / (2 * a)
            root2 = (-b - np.sqrt(delta)) / (2 * a)
            candidates = [root1, root2]
            between = [r for r in candidates if min(mu1, mu2) < r < max(mu1, mu2)]
            if len(between) == 1:
                return between[0]
            elif len(between) == 2:
                return min(between)
            else:
                print("compute_intersection: bad case, fallback to midpoint")
                return (mu1 + mu2) / 2.0

    def find_uncertainty_window(mu1, sigma1, pi1, mu2, sigma2, pi2, epsilon=0.05, grid_size=1000):
        sigma_avg = np.sqrt((sigma1 ** 2 + sigma2 ** 2) / 2.0)
        lo = max(0.0, min(mu1, mu2) - sigma_avg)
        hi = min(1.0, max(mu1, mu2) + sigma_avg)

        xs = np.linspace(lo, hi, grid_size)
        c1 = pi1 / (np.sqrt(2 * np.pi) * sigma1)
        c2 = pi2 / (np.sqrt(2 * np.pi) * sigma2)
        f1_vals = c1 * np.exp(-0.5 * ((xs - mu1) / sigma1) ** 2)
        f2_vals = c2 * np.exp(-0.5 * ((xs - mu2) / sigma2) ** 2)
        post1 = f1_vals / (f1_vals + f2_vals)

        lower_target = 0.5 - epsilon
        upper_target = 0.5 + epsilon

        idx_L = int(np.argmin(np.abs(post1 - lower_target)))
        idx_U = int(np.argmin(np.abs(post1 - upper_target)))

        l = xs[idx_L]
        u = xs[idx_U]

        print(f"L: {L:.4f}, U: {U:.4f}")
        return min(l, u), max(l, u)

    for i in range(len(active_sorted) - 1):
        idx1 = active_sorted[i]
        idx2 = active_sorted[i + 1]
        mu1, sigma1, w1 = means[idx1], stds[idx1], weights[idx1]
        mu2, sigma2, w2 = means[idx2], stds[idx2], weights[idx2]

        x0 = compute_intersection(idx1, idx2)

        sigma_avg = np.sqrt((sigma1 ** 2 + sigma2 ** 2) / 2.0)
        L0 = max(x0 - sigma_avg, min(mu1, mu2))
        U0 = min(x0 + sigma_avg, max(mu1, mu2))

        count_between = 0
        for seg in training_set:
            start, end = seg['start'], seg['end']
            for j in range(start, end + 1):
                if j < 0 or j >= len(scores):
                    continue
                s = scores[j]
                if L0 < s < U0:
                    count_between += 1
        print(f"窗口 [{L0:.4f}, {U0:.4f}] 内样本数: {count_between}")

        if count_between == 0:
            print("在交点 ± 平均σ范围内无任何训练样本，视作单峰合并")
            return 1, [-1]

        epsilon = 0.05
        L, U = find_uncertainty_window(mu1, sigma1, w1, mu2, sigma2, w2, epsilon=epsilon, grid_size=2000)

        uncertain_idxs = []
        for seg in training_set:
            start, end = seg['start'], seg['end']
            for j in range(start, end + 1):
                if len(scores) > j >= 0 == flags[j]:
                    x = scores[j]
                    if L <= x <= U:
                        uncertain_idxs.append(j)
        uncertain_idxs = sorted(set(uncertain_idxs))

        if len(uncertain_idxs) > MAX_ONCE:
            epsilon = 0.02
            L2, U2 = find_uncertainty_window(mu1, sigma1, w1, mu2, sigma2, w2, epsilon=epsilon, grid_size=2000)
            uncertain_idxs = []
            for seg in training_set:
                start, end = seg['start'], seg['end']
                for j in range(start, end + 1):
                    if len(scores) > j >= 0 == flags[j]:
                        x = scores[j]
                        if L2 <= x <= U2:
                            uncertain_idxs.append(j)
            uncertain_idxs = sorted(set(uncertain_idxs))

        if len(uncertain_idxs) < 10:
            print(111)
            epsilon = 0.1
            L3, U3 = find_uncertainty_window(mu1, sigma1, w1, mu2, sigma2, w2, epsilon=epsilon, grid_size=2000)
            uncertain_idxs = []
            for seg in training_set:
                start, end = seg['start'], seg['end']
                for j in range(start, end + 1):
                    if len(scores) > j >= 0 == flags[j]:
                        x = scores[j]
                        if L3 <= x <= U3:
                            uncertain_idxs.append(j)
            uncertain_idxs = sorted(set(uncertain_idxs))

        if len(uncertain_idxs) > MAX_ONCE:
            uncertain_idxs.sort(key=lambda idx: abs(scores[idx] - x0))
            uncertain_idxs = uncertain_idxs[:MAX_ONCE]

        if uncertain_idxs:
            print(f"后验 ∈ 筛得 {len(uncertain_idxs)} 个待判定样本")
            return 0, uncertain_idxs

    return 0, []
