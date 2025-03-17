# first line: 139
@memory.cache
def cached_fit(data_tuple, pattern_length):
    clf_off = NORMA(pattern_length=pattern_length, nm_size=3 * pattern_length,
                    percentage_sel=1, normalize='z-norm')
    clf_off.fit(np.array(data_tuple))
    return np.array(clf_off.decision_scores_)
