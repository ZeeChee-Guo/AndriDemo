from . import detect_anomalies_bp
from flask import request
from utils.ml import (lof_with_hyperparameters, lof_without_hyperparameters,
                      i_forest_without_hyperparameters, i_forest_with_hyperparameters)


@detect_anomalies_bp.route('i_forest', methods=['POST'])
def i_forest():
    json_data = request.get_json()
    data = json_data.get('data')
    flags = json_data.get('flags')
    areas = json_data.get('areas')
    if json_data.get('hyperparameters') == 0:
        return i_forest_without_hyperparameters(data, flags, areas)
    else:
        n_estimators = int(json_data.get('n_estimator', 100))
        contamination = float(json_data.get('contamination', 0.1))
        max_samples = int(json_data.get('maxSamples', 256))
        return i_forest_with_hyperparameters(data, flags, areas, n_estimators, contamination, max_samples)



@detect_anomalies_bp.route('LOF', methods=['POST'])
def lof():
    json_data = request.get_json()
    data = json_data.get('data')
    flags = json_data.get('flags')
    areas = json_data.get('areas')
    if json_data.get('hyperparameters') == 0:
        return lof_without_hyperparameters(data, flags, areas)
    else:
        n_Neighbors = int(json_data.get('nNeighbors', 20))
        contamination = float(json_data.get('contamination', 0.1))
        metric = json_data.get('metric', 'Euclidean')
        algorithm = json_data.get('algorithm', 'auto')
        leaf_size = int(json_data.get('leafSize', 30))
        p_Value = float(json_data.get('pValue', 2))

        return lof_with_hyperparameters(data, flags, areas, n_Neighbors, contamination, metric, algorithm, leaf_size,
                                        p_Value)
