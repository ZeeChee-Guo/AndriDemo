import numpy as np
from joblib import Memory

from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import BayesianGaussianMixture
import warnings
import pandas as pd
from TSB_UAD.models.damp import DAMP
from algorithms.andri.util_andri import find_length
from TSB_UAD.models.sand import SAND
from algorithms.andri.andri import AnDri
from TSB_UAD.utils.slidingWindows import find_length as find_sand_length

memory = Memory(location="./norma_cache", verbose=0)


@memory.cache
def compute_global_scores(data_arr, training_set, param_list):
    try:
        from algorithms.norma.norma import NORMA
    except ModuleNotFoundError:
        raise ImportError(
            "The NORMA module is proprietary and not included in this repository.\n"
            "Please request `norma.py` from the author and place it at: algorithms/norma/norma.py"
        )

    pattern_length = param_list.get('patternLength') or find_length(data_arr)

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
        segments.append({'start': start_idx, 'end': end_idx, 'is_train': curr_flag})
        idx += 1

    scores_full = np.zeros(N)
    all_nm = []
    all_nm_weights = []

    for seg in segments:
        seg_data = data_arr[seg['start']:seg['end'] + 1]
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
                padded_scores = np.pad(padded_scores, (0, seg_data.shape[0] - padded_scores.shape[0]), 'edge')
        else:
            padded_scores = np.zeros(len(seg_data))

        scores_full[seg['start']:seg['end'] + 1] = padded_scores

        if seg['is_train']:
            all_nm.extend(clf.normalmodel[0])
            all_nm_weights.extend(clf.normalmodel[1])


    return pattern_length, scores_full, all_nm, all_nm_weights


@memory.cache
def compute_training_set_sand_scores(data_arr, training_set, param_list):
    batch_size = param_list.get('batch_size') or None
    init_length = param_list.get('init_length') or None
    slidingWindow = param_list.get('slidingWindow') or find_sand_length(data_arr)

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
        segments.append({'start': start_idx, 'end': end_idx, 'is_train': curr_flag})
        idx += 1

    scores_full = np.zeros(N)
    all_clusters = []
    all_weights = []

    for seg in segments:
        seg_data = data_arr[seg['start']:seg['end'] + 1]
        if len(seg_data) == 0:
            continue
        clf = SAND(
            pattern_length=slidingWindow,
            subsequence_length=4 * slidingWindow,
        )
        clf.fit(seg_data, init_length=init_length, batch_size=batch_size, overlaping_rate=int(1.5 * slidingWindow))
        scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(clf.decision_scores_.reshape(-1, 1)).ravel()
        scores_full[seg['start']:seg['end'] + 1] = scores

        if seg['is_train']:
            clusters = [c[0] for c in clf.clusters]
            weights = clf.weights
            all_clusters.extend(clusters)
            all_weights.extend(weights)

    return (slidingWindow, scores_full, [c.tolist() for c in all_clusters], [float(w) for w in all_weights],
            init_length, batch_size)


@memory.cache
def compute_training_set_damp_scores(data_arr, training_set, param_list):
    m = param_list.get('m') or find_length(data_arr)
    x_lag = param_list.get('xLag') or 2**int(np.ceil(np.log2( 8*m )))

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
        segments.append({'start': start_idx, 'end': end_idx, 'is_train': curr_flag})
        idx += 1

    scores_full = np.zeros(N)

    for seg in segments:
        seg_data = data_arr[seg['start']:seg['end'] + 1]
        if len(seg_data) == 0:
            continue
        clf = DAMP(m=m, sp_index=m + 1, x_lag=x_lag)
        clf.fit(seg_data)
        raw_scores = clf.decision_scores_
        pad = (m - 1) // 2

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

        scores_full[seg['start']:seg['end'] + 1] = padded_scores

    return m, scores_full, [], [], x_lag


@memory.cache
def compute_andri_scores(data_arr, param_list):
    slidingWindow = param_list.get('slidingWindow') or find_length(data_arr)
    max_W = param_list.get('max_W') or 20
    k_adj = param_list.get('Kadj') or 1

    N = len(data_arr)
    clf = AnDri(
        slidingWindow, 'zero-mean', linkage_method='ward', th_reverse=5, kadj=k_adj,
        nm_len=2, overlap=0, max_W=max_W, eta=1
    )
    clf.fit(
        data_arr,
        y=None,
        online=False,
        training_len=int(0.2*N),
        stepwise=True,
        min_size=0.025
    )
    score = clf.scores
    score = MinMaxScaler(feature_range=(0, 1)).fit_transform(score.reshape(-1, 1)).ravel()
    if len(score) < N:
        score = np.append(score, np.ones(N - len(score)) * np.mean(score))
    all_scores = score

    # NM subseqs
    all_NM_subseqs = [nm.subseq.tolist() for nm in clf.NMs]

    # NM indices
    cl_s = clf.cl_s
    if len(cl_s) > N:
        cl_s = cl_s[:N]
    elif len(cl_s) < N:
        cl_s = np.append(cl_s, np.ones(N - len(cl_s)) * cl_s[-1])
    all_nm_indices = cl_s.astype(int)

    return slidingWindow, all_scores, all_nm_indices, all_NM_subseqs, k_adj, max_W


def andri_scoring(data, flags, param_list):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores, nm_indices, nms, k_adj, max_w = compute_andri_scores(data_arr, param_list)

    return {
        'global_scores': global_scores.tolist(),
        'pattern_length': int(pattern_length),
        'flags': flags.astype(int).tolist(),
        'nms': nms,
        'nm_indices': nm_indices.tolist(),
        'k_adj': k_adj,
        'max_w': max_w,
    }


def sand_scoring(data, flags, training_set, params):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores, nms, weights, init_length, batch_size = compute_training_set_sand_scores(
        data_arr, training_set, params
    )

    return {
        'global_scores': list(global_scores),
        'pattern_length': int(pattern_length),
        'flags': list(map(int, flags)),
        'nms': nms,
        'weights': weights,
        'init_length': init_length,
        'batch_size': batch_size
    }


def norm_a_scoring(data, flags, training_set, params):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores, nms, weights = compute_global_scores(data_arr, training_set, params)

    return {
        'global_scores': list(global_scores),
        'pattern_length': int(pattern_length),
        'flags': list(map(int, flags)),
        'nms': nms,
        'weights': weights,
    }


def damp_scoring(data, flags, training_set, params):
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    flags = np.array(flags)
    flags[np.isin(flags, [1])] = 1

    data_arr = np.array(data, dtype=float).flatten()
    pattern_length, global_scores, nms, weights, x_lag = compute_training_set_damp_scores(data_arr, training_set, params)

    return {
        'global_scores': list(global_scores),
        'pattern_length': int(pattern_length),
        'flags': list(map(int, flags)),
        'nms': nms,
        'weights': weights,
        'x_lag': x_lag
    }



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

    if scores_sel.shape[0] < 2:
        print("No enough（flag==1），cannot fit BGMM。")
        return (
            1,
            [-1],  # intersect
            [1],  # means
            [0],  # stds
            1,  # mean_active
            0,  # std_active
            []  # weights
        )


    bgmm = BayesianGaussianMixture(
        n_components=2,
        weight_concentration_prior_type='dirichlet_process',
        random_state=0
    )
    bgmm.fit(scores_sel)

    weight_threshold = 0.05
    weights = bgmm.weights_
    means = bgmm.means_.flatten()
    covariances = bgmm.covariances_.flatten()
    stds = np.sqrt(covariances)


    labels = bgmm.predict(scores_sel)
    active_idx = np.where(weights > weight_threshold)[0]

    mask_active_points = np.isin(labels, active_idx)
    scores_active = scores_sel[mask_active_points].flatten()


    mean_active = float(np.mean(scores_active)) if scores_active.size > 0 else 0.0
    std_active  = float(np.std(scores_active))  if scores_active.size > 0 else 0.0


    if len(active_idx) < 2:
        print('剩下不到两个活跃成分:')
        for idx in active_idx:
            print(f"  mean: {means[idx]:.4f}, std: {stds[idx]:.4f}, weight: {weights[idx]:.4f}")
        return (
            1,               # stable
            [-1],            # intersect
            means.tolist(),
            stds.tolist(),
            mean_active,     #
            std_active,
            weights.tolist()
        )


    def compute_intersection(idx1, idx2):
        mu1, sigma1, w1 = means[idx1], stds[idx1], weights[idx1]
        mu2, sigma2, w2 = means[idx2], stds[idx2], weights[idx2]
        a = 1.0 / (sigma2 ** 2) - 1.0 / (sigma1 ** 2)
        b = (-2 * mu2) / (sigma2 ** 2) + (2 * mu1) / (sigma1 ** 2)
        C = 2 * (np.log(w2 / sigma2) - np.log(w1 / sigma1))
        c = (mu2 ** 2) / (sigma2 ** 2) - (mu1 ** 2) / (sigma1 ** 2) - C
        if abs(a) < 1e-12:
            return -c / b if abs(b) > 1e-12 else (mu1 + mu2) / 2.0
        delta = b ** 2 - 4 * a * c
        if delta < 0:
            return (mu1 + mu2) / 2.0
        roots = [(-b + np.sqrt(delta)) / (2 * a), (-b - np.sqrt(delta)) / (2 * a)]
        between = [r for r in roots if min(mu1, mu2) < r < max(mu1, mu2)]
        return between[0] if len(between) == 1 else (min(between) if between else (mu1 + mu2) / 2.0)


    def find_uncertainty_window(mu1, sigma1, pi1, mu2, sigma2, pi2, epsilon=0.05, grid_size=1000):
        sigma_avg = np.sqrt((sigma1 ** 2 + sigma2 ** 2) / 2.0)
        lo = max(0.0, min(mu1, mu2) - sigma_avg)
        hi = min(1.0, max(mu1, mu2) + sigma_avg)
        xs = np.linspace(lo, hi, grid_size)
        c1 = pi1 / (np.sqrt(2 * np.pi) * sigma1)
        c2 = pi2 / (np.sqrt(2 * np.pi) * sigma2)
        f1 = c1 * np.exp(-0.5 * ((xs - mu1) / sigma1) ** 2)
        f2 = c2 * np.exp(-0.5 * ((xs - mu2) / sigma2) ** 2)
        post1 = f1 / (f1 + f2)
        target_low, target_high = 0.5 - epsilon, 0.5 + epsilon
        idx_L = int(np.argmin(np.abs(post1 - target_low)))
        idx_U = int(np.argmin(np.abs(post1 - target_high)))
        return min(xs[idx_L], xs[idx_U]), max(xs[idx_L], xs[idx_U])

    active_sorted = sorted(active_idx, key=lambda i: means[i])
    MAX_ONCE = 20

    for i in range(len(active_sorted) - 1):
        idx1, idx2 = active_sorted[i], active_sorted[i + 1]
        mu1, sigma1, pi1 = means[idx1], stds[idx1], weights[idx1]
        mu2, sigma2, pi2 = means[idx2], stds[idx2], weights[idx2]
        x0 = compute_intersection(idx1, idx2)
        sigma_avg = np.sqrt((sigma1 ** 2 + sigma2 ** 2) / 2.0)
        L0 = max(x0 - sigma_avg, min(mu1, mu2))
        U0 = min(x0 + sigma_avg, max(mu1, mu2))
        count_between = sum(
            1 for seg in training_set for j in range(seg['start'], seg['end']+1)
            if len(scores) > j >= 0 == flags[j] and L0 < scores[j] < U0
        )
        if count_between == 0:
            return 1, [-1], means.tolist(), stds.tolist(), mean_active, std_active, weights.tolist()

        L, U = find_uncertainty_window(mu1, sigma1, pi1, mu2, sigma2, pi2)
        uncertain = [j for seg in training_set for j in range(seg['start'], seg['end']+1)
                     if len(scores) > j >= 0 == flags[j] and L <= scores[j] <= U]
        uncertain = sorted(set(uncertain))
        if len(uncertain) > MAX_ONCE:
            L2, U2 = find_uncertainty_window(mu1, sigma1, pi1, mu2, sigma2, pi2, epsilon=0.02, grid_size=2000)
            uncertain = [j for seg in training_set for j in range(seg['start'], seg['end']+1)
                         if len(scores) > j >= 0 == flags[j] and L2 <= scores[j] <= U2]
            uncertain = sorted(set(uncertain))
        if len(uncertain) < 10:
            L3, U3 = find_uncertainty_window(mu1, sigma1, pi1, mu2, sigma2, pi2, epsilon=0.2, grid_size=2000)
            uncertain = [j for seg in training_set for j in range(seg['start'], seg['end']+1)
                         if len(scores) > j >= 0 == flags[j] and L3 <= scores[j] <= U3]
            uncertain = sorted(set(uncertain))
        if len(uncertain) > MAX_ONCE:
            uncertain.sort(key=lambda j: abs(scores[j] - x0))
            uncertain = uncertain[:MAX_ONCE]
        if uncertain:
            return 0, uncertain, means.tolist(), stds.tolist(), mean_active, std_active, weights.tolist()

    return 1, [-1], means.tolist(), stds.tolist(), mean_active, std_active, weights.tolist()

