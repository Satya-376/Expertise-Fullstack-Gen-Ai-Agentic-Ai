from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello Flask!</p>"


app.run(host='127.0.0.1',port=81)

