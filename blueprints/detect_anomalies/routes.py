from . import detect_anomalies_bp
from flask import request
from utils.ml import  norm_a_scoring


@detect_anomalies_bp.route('norm_a', methods=['POST'])
def norm_a():
    json_data = request.get_json()
    data = json_data.get('data')
    flags = json_data.get('flags')
    areas = json_data.get('areas')

    return norm_a_scoring(data, flags, areas)
