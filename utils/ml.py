import numpy as np
from joblib import Memory

from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture
import warnings
import pandas as pd
from algotithms.norma.norma import NORMA
from TSB_UAD.models.damp import DAMP
from algotithms.andri.util_a2d2 import find_length
from TSB_UAD.models.sand import SAND
from algotithms.andri.a2d2 import A2D2
from TSB_UAD.utils.slidingWindows import find_length as find_sand_length

memory = Memory(location="./norma_cache", verbose=0)


@memory.cache
def compute_global_scores(data_arr, training_set, pattern_length=None):
    if pattern_length is None:
        pattern_length = find_length(data_arr)

    N = len(data_arr)
    is_train = np.zeros(N, dtype=bool)
    for seg in training_set:
        is_train[seg['start']:seg['end']+1] = True

    segments = []
    idx = 0
    while idx < N:
        start_idx = idx
        curr_flag = is_train[idx]
        while idx + 1 < N and is_train[idx + 1] == curr_flag:
            idx += 1
        end_idx = idx
        segments.append({'start': start_idx, 'end': end_idx, 'is_train': curr_flag})
        idx += 1

    scores_full = np.zeros(N)
    all_nm = []
    all_nm_weights = []

    for seg in segments:
        seg_data = data_arr[seg['start']:seg['end']+1]
        if len(seg_data) == 0:
            continue
        clf = NORMA(
            pattern_length=pattern_length,
            nm_size=3 * pattern_length,
            percentage_sel=1,
            normalize='z-norm'
        )
        clf.fit(seg_data)
        raw_scores = clf.decision_scores_
        pad = (pattern_length - 1) // 2

        if raw_scores.size > 0:
            norm_scores = MinMaxScaler(feature_range=(0, 1)) \
                .fit_transform(raw_scores.reshape(-1, 1)).ravel()
            left_pad = np.full(pad, norm_scores[0])
            right_pad = np.full(pad, norm_scores[-1])
            padded_scores = np.concatenate([left_pad, norm_scores, right_pad])
            if padded_scores.shape[0] > seg_data.shape[0]:
                padded_scores = padded_scores[:seg_data.shape[0]]
            elif padded_scores.shape[0] < seg_data.shape[0]:
                padded_scores = np.pad(padded_scores, (0, seg_data.shape[0]-padded_scores.shape[0]), 'edge')
        else:
            padded_scores = np.zeros(len(seg_data))

        scores_full[seg['start']:seg['end']+1] = padded_scores

        if seg['is_train']:
            all_nm.append(clf.normalmodel[0])
            all_nm_weights.append(clf.normalmodel[1])

    return pattern_length, scores_full, all_nm, all_nm_weights


@memory.cache
def compute_training_set_sand_scores(data_arr, training_set, slidingWindow=None):
    if slidingWindow is None:
        slidingWindow = find_sand_length(data_arr)

    N = len(data_arr)
    is_train = np.zeros(N, dtype=bool)
    for seg in training_set:
        is_train[seg['start']:seg['end']+1] = True

    segments = []
    idx = 0
    while idx < N:
        start_idx = idx
        curr_flag = is_train[idx]
        while idx + 1 < N and is_train[idx + 1] == curr_flag:
            idx += 1
        end_idx = idx
        segments.append({'start': start_idx, 'end': end_idx, 'is_train': curr_flag})
        idx += 1

    scores_full = np.zeros(N)
    all_clusters = []
    all_weights = []

    for seg in segments:
        seg_data = data_arr[seg['start']:seg['end']+1]
        if len(seg_data) == 0:
            continue
        clf = SAND(
            pattern_length=slidingWindow,
            subsequence_length=4 * slidingWindow,
        )
        clf.fit(seg_data, overlaping_rate=int(1.5 * slidingWindow))
        scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(clf.decision_scores_.reshape(-1, 1)).ravel()
        scores_full[seg['start']:seg['end']+1] = scores

        if seg['is_train']:
            clusters = [c[0] for c in clf.clusters]
            weights = clf.weights
            all_clusters.append(clusters)
            all_weights.append(weights)

    return slidingWindow, scores_full, all_clusters, all_weights


@memory.cache
def compute_training_set_damp_scores(data_arr, training_set, slidingWindow=None):
    if slidingWindow is None:
        slidingWindow = find_sand_length(data_arr)

    N = len(data_arr)
    is_train = np.zeros(N, dtype=bool)
    for seg in training_set:
        is_train[seg['start']:seg['end']+1] = True

    segments = []
    idx = 0
    while idx < N:
        start_idx = idx
        curr_flag = is_train[idx]
        while idx + 1 < N and is_train[idx + 1] == curr_flag:
            idx += 1
        end_idx = idx
        segments.append({'start': start_idx, 'end': end_idx, 'is_train': curr_flag})
        idx += 1

    scores_full = np.zeros(N)

    for seg in segments:
        seg_data = data_arr[seg['start']:seg['end']+1]
        if len(seg_data) == 0:
            continue
        clf = DAMP(m=slidingWindow, sp_index=slidingWindow + 1)
        clf.fit(seg_data)
        raw_scores = clf.decision_scores_
        pad = (slidingWindow - 1) // 2

        if raw_scores.size > 0:
            damp_scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(raw_scores.reshape(-1, 1)).ravel()
            left_pad = np.full(pad, damp_scores[0])
            right_pad = np.full(pad, damp_scores[-1])
            padded_scores = np.concatenate([left_pad, damp_scores, right_pad])
            if padded_scores.shape[0] > seg_data.shape[0]:
                padded_scores = padded_scores[:seg_data.shape[0]]
            elif padded_scores.shape[0] < seg_data.shape[0]:
                padded_scores = np.pad(padded_scores, (0, seg_data.shape[0] - padded_scores.shape[0]), 'edge')
        else:
            padded_scores = np.zeros(len(seg_data))

        scores_full[seg['start']:seg['end']+1] = padded_scores

    return slidingWindow, scores_full, [], []


@memory.cache
def compute_training_set_andri_scores(data_arr, training_set, param_list):
    slidingWindow = param_list.get('slidingWindow', find_sand_length(data_arr))
    max_W = param_list.get('max_W', 20)
    min_size = param_list.get('min_size', 0.01)

    N = len(data_arr)
    is_train = np.zeros(N, dtype=bool)
    for seg in training_set:
        is_train[seg['start']:seg['end'] + 1] = True

    segments = []
    idx = 0
    while idx < N:
        start_idx = idx
        curr_flag = is_train[idx]
        while idx + 1 < N and is_train[idx + 1] == curr_flag:
            idx += 1
        end_idx = idx
        segments.append({'start': start_idx, 'end': end_idx})
        idx += 1

    all_scores = np.zeros(N)
    all_nm_indices = np.zeros(N, dtype=int)
    all_NM_subseqs = []
    nm_offset = 0

    for seg in segments:
        seg_data = data_arr[seg['start']:seg['end'] + 1]
        clf = A2D2(
            slidingWindow, 'zero-mean', linkage_method='ward', th_reverse=5, kadj=1,
            nm_len=2, overlap=0, max_W=max_W, eta=1
        )
        clf.fit(
            seg_data,
            y=None,
            online=False,
            training=True,
            training_len=len(seg_data),
            stump=False,
            stepwise=True,
            min_size=min_size
        )
        score = clf.scores
        score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
        if len(score) < len(seg_data):
            score = np.append(score, np.ones(len(seg_data) - len(score)) * np.mean(score))
        all_scores[seg['start']:seg['end'] + 1] = score

        NM_subseqs = [nm.subseq.tolist() for nm in clf.NMs]
        all_NM_subseqs.extend(NM_subseqs)
        cl_s = clf.cl_s

        if len(cl_s) > len(seg_data):
            cl_s = cl_s[:len(seg_data)]
        elif len(cl_s) < len(seg_data):
            cl_s = np.append(cl_s, np.ones(len(seg_data) - len(cl_s)) * cl_s[-1])
        all_nm_indices[seg['start']:seg['end'] + 1] = cl_s + nm_offset
        nm_offset += len(clf.NMs)

    return slidingWindow, all_scores, all_nm_indices, all_NM_subseqs


def andri_scoring(data, flags, training_set, param_list):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1, 3])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores, nm_indices, nms = compute_training_set_andri_scores(data_arr, training_set,
                                                                                       param_list)

    return {
        'global_scores': global_scores.tolist(),
        'pattern_length': int(pattern_length),
        'flags': flags.astype(int).tolist(),
        'nms': nms,
        'nm_indices': nm_indices.tolist()
    }


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


def norm_a_scoring(data, flags, training_set, pattern_length=None):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1, 3])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores, nms, weights = compute_global_scores(data_arr, training_set, pattern_length)

    return {
        'global_scores': list(global_scores),
        'pattern_length': int(pattern_length),
        'flags': list(map(int, flags)),
        'nms': nms,
        'weights': weights,
    }


def damp_scoring(data, flags, training_set, pattern_length=None):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1, 3])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores, nms, weights = compute_training_set_damp_scores(data_arr, training_set,
                                                                                   pattern_length)

    return {
        'global_scores': list(global_scores),
        'pattern_length': int(pattern_length),
        'flags': list(map(int, flags)),
        'nms': nms,
        'weights': weights,
    }


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
        n_components=2,
        weight_concentration_prior_type='dirichlet_process',
        random_state=0
    )
    bgmm.fit(scores_sel)

    weight_threshold = 0.01
    weights = bgmm.weights_
    means = bgmm.means_.flatten()
    covariances = bgmm.covariances_.flatten()
    stds = np.sqrt(covariances)

    labels = bgmm.predict(scores_sel)
    active_idx = np.where(weights > weight_threshold)[0]
    mask_active_points = np.isin(labels, active_idx)
    scores_active = scores_sel[mask_active_points].flatten()
    std_all_active = np.std(scores_active)

    if len(active_idx) < 2:
        print('剩下不到两个活跃成分:')
        for idx in active_idx:
            print(f"  mean: {means[idx]:.4f}, std: {stds[idx]:.4f}, weight: {weights[idx]:.4f}")
        return 1, [-1], means.tolist(), stds.tolist(), std_all_active, weights.tolist()

    for idx in active_idx:
        print(f"  mean: {means[idx]:.4f}, std: {stds[idx]:.4f}, weight: {weights[idx]:.4f}")

    print(f"所有有效成分内的点的标准差: {std_all_active:.4e}")

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

        print(f"L: {l:.4f}, U: {u:.4f}")
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
                if L0 < s < U0 and flags[j] == 0:
                    count_between += 1
        print(f"窗口 [{L0:.4f}, {U0:.4f}] 内样本数: {count_between}")

        if count_between == 0:
            print("在交点 ± 平均σ范围内无任何训练样本，视作单峰合并")
            return 1, [-1], means.tolist(), stds.tolist(), std_all_active, weights.tolist()

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
            epsilon = 0.2
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
            return 0, uncertain_idxs, means.tolist(), stds.tolist(), std_all_active, weights.tolist()

    return 1, [-1], means.tolist(), stds.tolist(), std_all_active, weights.tolist()
