# first line: 16
@memory.cache
def compute_global_scores(data_arr):
    """Fit NORMA and return pattern_length and normalized scores with padding."""
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
