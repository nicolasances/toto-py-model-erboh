from flask import Flask
from controller.controller import ModelController

app = Flask(__name__)

model_controller = ModelController('erboh', app)

# TODO: the model controller should be called with the methods to call for train, score, predict_batch, etc....
# model_controller.set_predict_batch(predict_batch)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)