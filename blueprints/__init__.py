from blueprints.detect_anomalies import detect_anomalies_bp

def register_blueprints(app):
    app.register_blueprint(detect_anomalies_bp)