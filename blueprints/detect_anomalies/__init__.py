from flask import Blueprint

detect_anomalies_bp = Blueprint('detect_anomalies', __name__, url_prefix='/detect-anomalies')

from . import routes