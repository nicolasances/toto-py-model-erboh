from flask import Flask
from controller.controller import ModelController

# Microservice name
ms_name = 'model-erboh'
model_name = 'erboh'

app = Flask(__name__)

model_controller = ModelController(model_name, app)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)