from flask import Flask, jsonify, request, Response
from toto_pubsub.consumer import TotoEventConsumer
from predict.batch import predict as predict_batch
from predict.single import predict as predict_single

print('app:first line)

def create_app(): 

    print('app:creating app')

    # Microservice name
    ms_name = 'model-erboh'

    app = Flask(__name__)

    # Event Consumers
    TotoEventConsumer(ms_name, ['erbohBatchInferenceRequested', 'erbohPredictionRequested'], [predict_batch, predict_single])

    # APIs
    @app.route('/')
    def smoke():
        return jsonify({
            "api": "erboh",
            "status": "running"
        })

    print('app:returning app')

    return app

print("app: calling create()")
app = create_app()
print("app: created")

if __name__ == '__main__':
    print("app: main")
    app.run(host='0.0.0.0', port=8080)
    print("app: post run")