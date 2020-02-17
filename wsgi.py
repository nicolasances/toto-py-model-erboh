from app import app
from toto_pubsub.consumer import TotoEventConsumer
from predict.batch import predict as predict_batch
from predict.single import predict as predict_single

if __name__ == "__main__":
    app.run()

    # Microservice name
    ms_name = 'model-erboh'

    # Event Consumers
    TotoEventConsumer(ms_name, ['erbohBatchInferenceRequested', 'erbohPredictionRequested'], [predict_batch, predict_single])