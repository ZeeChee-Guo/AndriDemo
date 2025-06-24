from . import detect_anomalies_bp
from flask import request, jsonify
from utils.ml import norm_a_scoring, fit_user_labels, sand_scoring, damp_scoring, andri_scoring
from utils.find_similar_patterns import find_similar_patterns, find_similar_anomaly_seq
import os, json
import numpy as np




@detect_anomalies_bp.route('sand', methods=['POST'])
def sand():
    cached_result = load_cache(CACHE_FILE_PATH2)
    if cached_result is not None:
        return jsonify(cached_result)

    json_data = request.get_json()
    data = json_data.get('data')
    flags = json_data.get('flags')
    training_set = json_data.get('areas')
    params = json_data.get('hyperParameters')
    result = sand_scoring(data, flags, training_set, params)

    if isinstance(result, dict):
        # save_cache(CACHE_FILE_PATH2, result)
        return jsonify(result)
    else:
        return result



@detect_anomalies_bp.route('damp', methods=['POST'])
def damp():
    cached_result = load_cache(CACHE_FILE_PATH3)
    if cached_result is not None:
        return jsonify(cached_result)

    json_data = request.get_json()
    data = json_data.get('data')
    flags = json_data.get('flags')
    training_set = json_data.get('areas')
    params = json_data.get('hyperParameters')
    result = damp_scoring(data, flags, training_set, params)

    if isinstance(result, dict):
        # save_cache(CACHE_FILE_PATH3, result)
        return jsonify(result)
    else:
        return result



@detect_anomalies_bp.route('norm_a', methods=['POST'])
def norm_a():
    cached_result = load_cache(CACHE_FILE_PATH1)
    if cached_result is not None:
        return jsonify(cached_result)

    json_data = request.get_json()
    data = json_data.get('data')
    flags = json_data.get('flags')
    training_set = json_data.get('areas')
    params = json_data.get('hyperParameters')
    result = norm_a_scoring(data, flags, training_set, params)


    if isinstance(result, dict):
        # save_cache(CACHE_FILE_PATH1, result)
        return jsonify(result)
    else:
        return result


@detect_anomalies_bp.route('andri', methods=['POST'])
def andri():
    cached_result = load_cache(CACHE_FILE_PATH4)
    if cached_result is not None:
        return jsonify(cached_result)

    json_data = request.get_json()
    data = json_data.get('data')
    flags = json_data.get('flags')
    params = json_data.get('hyperParameters')
    result = andri_scoring(data, flags, params)

    if isinstance(result, dict):
        save_cache(CACHE_FILE_PATH4, result)
        return jsonify(result)
    else:
        return result


CACHE_FILE_PATH1 = 'norm_a_cache.json'
CACHE_FILE_PATH2 = 'sand_cache.json'
CACHE_FILE_PATH3 = 'damp_cache.json'
CACHE_FILE_PATH4 = 'andri_cache.json'


def load_cache(CACHE_FILE_PATH):
    if os.path.exists(CACHE_FILE_PATH):
        with open(CACHE_FILE_PATH, 'r') as f:
            return json.load(f)
    return None

def save_cache(CACHE_FILE_PATH,result):
    with open(CACHE_FILE_PATH, 'w') as f:
        json.dump(result, f)




@detect_anomalies_bp.route('find_similar_pattern', methods=['POST'])
def find_similar_pattern():
    json_data = request.get_json()
    original_data = json_data.get('original_seq')
    nms = json_data.get('norm_a_normal_patterns')
    nm = find_similar_patterns(original_data, nms)

    return jsonify(nm)



@detect_anomalies_bp.route('find_similar_anomaly_pattern', methods=['POST'])
def find_similar_anomaly_pattern():
    json_data = request.get_json()
    original_data = json_data.get('original_seq')
    nms = json_data.get('norm_a_normal_patterns')
    seq = find_similar_anomaly_seq(original_data, nms)

    return jsonify(seq)



@detect_anomalies_bp.route('norm_a_fit_user_labels', methods=['POST'])
def norm_a_fit_user_labels():
    json_data = request.get_json()
    scores = np.array(json_data.get('scores'))
    flags = np.array(json_data.get('flags'))
    training_set = json_data.get('training_set')

    stable, intersect, mu, sigma, active_mean, active_std, weights = fit_user_labels(scores, flags, training_set)

    return jsonify({'stable': stable, 'intersect': intersect, 'mu':mu, 'sigma': sigma,
                    'active_mean': active_mean, 'active_std': active_std,'weights': weights})
