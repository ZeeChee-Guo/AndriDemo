import json
import os
import tempfile

from . import detect_anomalies_bp
from flask import request, jsonify
from utils.ml import norm_a_scoring, fit_user_labels, sand_scoring, damp_scoring, andri_scoring
from utils.find_similar_patterns import find_similar_patterns, find_similar_anomaly_seq
import numpy as np




CACHE_DIR = os.environ.get('ANOMALY_CACHE_DIR', os.path.join(os.getcwd(), 'anomaly_cache'))
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}_result.json")

def _load_cache(name: str):
    path = _cache_path(name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_cache_atomic(name: str, data):
    # 原子写入，避免并发时产生半写文件
    path = _cache_path(name)
    fd, tmp_path = tempfile.mkstemp(dir=CACHE_DIR, prefix=f"{name}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)  # 原子替换
    finally:
        # 如果异常导致 tmp 没被替换，清理之
        if os.path.exists(tmp_path) and not os.path.samefile(tmp_path, path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def _return_cached_or_compute(cache_key: str, compute_fn):
    """如果有缓存就直接返回；否则计算并写入缓存。"""
    cached = _load_cache(cache_key)
    if cached is not None:
        return jsonify(cached)

    # 首次计算
    result = compute_fn()
    # 若 result 不是可 JSON 序列化对象，可在此进行转化
    _save_cache_atomic(cache_key, result)
    return jsonify(result)

# ----------------- 以下为四个接口 -----------------

@detect_anomalies_bp.route('sand', methods=['POST'])
def sand():
    json_data = request.get_json() or {}
    def _compute():
        data = json_data.get('data')
        flags = json_data.get('flags')
        training_set = json_data.get('areas')
        params = json_data.get('hyperParameters')
        baseline = json_data.get('baseline')
        return sand_scoring(data, flags, training_set, params, baseline)
    return _return_cached_or_compute("sand", _compute)

@detect_anomalies_bp.route('damp', methods=['POST'])
def damp():
    json_data = request.get_json() or {}
    def _compute():
        data = json_data.get('data')
        flags = json_data.get('flags')
        training_set = json_data.get('areas')
        params = json_data.get('hyperParameters')
        baseline = json_data.get('baseline')
        return damp_scoring(data, flags, training_set, params, baseline)
    return _return_cached_or_compute("damp", _compute)

@detect_anomalies_bp.route('norm_a', methods=['POST'])
def norm_a():
    json_data = request.get_json() or {}
    def _compute():
        data = json_data.get('data')
        flags = json_data.get('flags')
        training_set = json_data.get('areas')
        params = json_data.get('hyperParameters')
        baseline = json_data.get('baseline')
        return norm_a_scoring(data, flags, training_set, params, baseline)
    return _return_cached_or_compute("norm_a", _compute)

@detect_anomalies_bp.route('andri', methods=['POST'])
def andri():
    json_data = request.get_json() or {}
    def _compute():
        data = json_data.get('data')
        flags = json_data.get('flags')
        params = json_data.get('hyperParameters')
        baseline = json_data.get('baseline')
        return andri_scoring(data, flags, params, baseline)
    return _return_cached_or_compute("andri", _compute)


# @detect_anomalies_bp.route('sand', methods=['POST'])
# def sand():
#     json_data = request.get_json()
#     data = json_data.get('data')
#     flags = json_data.get('flags')
#     training_set = json_data.get('areas')
#     params = json_data.get('hyperParameters')
#     baseline = json_data.get('baseline')
#     result = sand_scoring(data, flags, training_set, params, baseline)
#
#     return result
#
#
#
# @detect_anomalies_bp.route('damp', methods=['POST'])
# def damp():
#     json_data = request.get_json()
#     data = json_data.get('data')
#     flags = json_data.get('flags')
#     training_set = json_data.get('areas')
#     params = json_data.get('hyperParameters')
#     baseline = json_data.get('baseline')
#     result = damp_scoring(data, flags, training_set, params, baseline)
#     return result
#
#
#
# @detect_anomalies_bp.route('norm_a', methods=['POST'])
# def norm_a():
#     json_data = request.get_json()
#     data = json_data.get('data')
#     flags = json_data.get('flags')
#     training_set = json_data.get('areas')
#     params = json_data.get('hyperParameters')
#     baseline = json_data.get('baseline')
#     result = norm_a_scoring(data, flags, training_set, params, baseline)
#
#     return result
#
#
# @detect_anomalies_bp.route('andri', methods=['POST'])
# def andri():
#     json_data = request.get_json()
#     data = json_data.get('data')
#     flags = json_data.get('flags')
#     params = json_data.get('hyperParameters')
#     baseline = json_data.get('baseline')
#     result = andri_scoring(data, flags, params, baseline)
#     return result




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
