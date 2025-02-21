from flask import Flask, render_template
from blueprints import register_blueprints

app = Flask(__name__)
register_blueprints(app)

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
