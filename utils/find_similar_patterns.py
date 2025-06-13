import stumpy
import numpy as np

def find_similar_patterns(original_seq, nms):
    best_pattern = None
    best_distance = float('inf')
    best_loc = None

    nm_len = len(nms[0]) if nms else 0
    seq_len = len(original_seq)

    if not nms or seq_len == 0 or nm_len == 0:
        return {'nm': None, 'loc': None, 'direction': None}

    if seq_len <= nm_len:
        query = original_seq
        for nm in nms:
            dist_profile = stumpy.mass(query, nm)
            min_dist = np.min(dist_profile)
            min_loc = np.argmin(dist_profile)
            if min_dist < best_distance:
                best_distance = min_dist
                best_pattern = nm
                best_loc = min_loc
    else:
        for nm in nms:
            dist_profile = stumpy.mass(nm, original_seq)
            min_dist = np.min(dist_profile)
            min_loc = np.argmin(dist_profile)
            if min_dist < best_distance:
                best_distance = min_dist
                best_pattern = nm
                best_loc = min_loc

    return {'nm': best_pattern, 'loc': int(best_loc) if best_loc is not None else None,}



def find_similar_anomaly_seq(original_seq, nms):
    best_pattern = None
    best_distance = float('inf')
    best_loc = None

    nm_len = len(nms[0]) if nms else 0
    seq_len = len(original_seq)

    if not nms or seq_len == 0 or nm_len == 0:
        return {'nm': None, 'loc': None, 'direction': None}

    if seq_len <= nm_len:
        query = original_seq
        for nm in nms:
            dist_profile = stumpy.mass(query, nm)
            min_dist = np.min(dist_profile)
            min_loc = np.argmin(dist_profile)
            if min_dist < best_distance:
                best_distance = min_dist
                best_pattern = nm
                best_loc = min_loc
    else:
        for nm in nms:
            dist_profile = stumpy.mass(nm, original_seq)
            min_dist = np.min(dist_profile)
            min_loc = np.argmin(dist_profile)
            if min_dist < best_distance:
                best_distance = min_dist
                best_pattern = nm
                best_loc = min_loc

    return {'nm': best_pattern, 'loc': int(best_loc) if best_loc is not None else None, }
