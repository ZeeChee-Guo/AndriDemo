from flask import Flask, render_template
from blueprints import register_blueprints

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
register_blueprints(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test1')
def test1():
    return render_template('test.html')


if __name__ == '__main__':
    app.run()
