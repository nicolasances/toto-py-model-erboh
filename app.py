from flask import Flask
from model.erboh import ERBOH

from totoml.controller import ModelController
from totoml.config import ControllerConfig

app = Flask(__name__)

model_controller = ModelController(ERBOH(), app, ControllerConfig(enable_batch_predictions_events=True, enable_single_prediction_events=True))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
