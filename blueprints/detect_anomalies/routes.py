from . import detect_anomalies_bp
from flask import request
from utils.ml import  norm_a


@detect_anomalies_bp.route('i_forest', methods=['POST'])
def i_forest():
    json_data = request.get_json()
    data = json_data.get('data')
    flags = json_data.get('flags')
    areas = json_data.get('areas')

    return norm_a(data, flags, areas)
