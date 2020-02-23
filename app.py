from flask import Flask, jsonify, request, Response
from toto_pubsub.consumer import TotoEventConsumer
from predict.batch import predict as predict_batch
from predict.single import predict as predict_single
from label.labeling import data_labeled
from dlg.storage import FileStorage

# Microservice name
ms_name = 'model-erboh'

FileStorage(ms_name, 1)

app = Flask(__name__)

# Event Consumers
TotoEventConsumer(ms_name, ['erboh-predict-batch', 'erboh-predict-single', 'erboh-data-labeled'], [predict_batch, predict_single, data_labeled])

# APIs
@app.route('/')
def smoke():
    return jsonify({
        "api": ms_name,
        "status": "running"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)