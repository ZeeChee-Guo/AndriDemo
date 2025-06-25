import os

from flask import Flask, render_template, jsonify
from blueprints import register_blueprints

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
register_blueprints(app)

@app.route('/')
def index():
    norma_path = os.path.join(app.root_path, 'algorithms', 'norma', 'norma.py')
    norma_exists = os.path.isfile(norma_path)
    return render_template('index.html', norma_exists=norma_exists)


if __name__ == '__main__':
    app.run()
