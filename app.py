from flask import Flask, jsonify, request, Response
from toto_pubsub.consumer import TotoEventConsumer
from dlg.storage import FileStorage
from controller.controller import ModelController

# Microservice name
ms_name = 'model-erboh'
model_name = 'erboh'

FileStorage(ms_name, 1)

model_controller = ModelController(model_name)

app = Flask(__name__)

# Event Consumers
TotoEventConsumer(ms_name, ['erboh-predict-batch', 'erboh-predict-single'], [model_controller.predict_batch, model_controller.predict_single])

# APIs
@app.route('/')
def smoke():
    return jsonify({
        "api": ms_name,
        "status": "running"
    })

@app.route('/train', methods=['POST'])
def train(): 
    try: 
        resp = jsonify(model_controller.train(request))
        resp.status_code = 200
        resp.headers['Content-Type'] = 'application/json'

        return resp

    except KeyError as e: 
        return jsonify({"code": 400, "message": str(e)})

@app.route('/score', methods=['GET'])
def metrics(): 
    try: 

        resp = jsonify(model_controller.score(request))
        resp.status_code = 200
        resp.headers['Content-Type'] = 'application/json'

        return resp

    except KeyError as e: 
        return jsonify({"code": 400, "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)