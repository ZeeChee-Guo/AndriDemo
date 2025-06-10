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



def find_similar_anomaly_seq(original_seq, flags, full_data):
    query = np.array([full_data[int(idx)] for idx in original_seq])

    seq_len = len(query)
    flags = np.array(flags)
    full_data = np.array(full_data)

    segments = []
    in_segment = False
    start = 0
    for i, f in enumerate(flags):
        if f == 1 and not in_segment:
            start = i
            in_segment = True
        elif f != 1 and in_segment:
            segments.append((start, i))
            in_segment = False
    if in_segment:
        segments.append((start, len(flags)))

    best_indices = None
    best_distance = float('inf')

    for seg_start, seg_end in segments:
        if (seg_end - seg_start) < 10:
            continue

        nm = full_data[seg_start:seg_end]
        nm_len = len(nm)

        if seq_len <= nm_len:
            dist_profile = stumpy.mass(query, nm)
            min_dist = np.min(dist_profile)
            candidate_indices = list(range(seg_start, seg_end))
        else:
            dist_profile = stumpy.mass(nm, query)
            min_dist = np.min(dist_profile)
            candidate_indices = list(range(seg_start, seg_end))

        if min_dist < best_distance:
            best_distance = min_dist
            best_indices = candidate_indices

    best_indices = best_indices if best_indices is not None else None
    return {'seq': best_indices}

