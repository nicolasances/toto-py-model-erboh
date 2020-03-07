from flask import Flask
from controller.controller import ModelController
from model.erboh import ModelDelegate

app = Flask(__name__)

model_controller = ModelController('erboh', app, ModelDelegate())
model_controller.predict_batch()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)