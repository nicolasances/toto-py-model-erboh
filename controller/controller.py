import datetime
from random import randint
from flask import Flask, jsonify, request, Response

from toto_pubsub.consumer import TotoEventConsumer
from toto_pubsub.publisher import TotoEventPublisher

from remote.totoml_registry import check_registry
from remote.gcpremote import init_champion_model

from processes.training import TrainingProcess
from processes.scoring import ScoreProcess
from processes.predict_single import predict as predict_single
from processes.predict_batch import predict as predict_batch

def cid(): 
    '''
    Generates a Toto-valid Correlation ID
    Example: 20200229160021694-09776
    '''
    datepart = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    randpart = str(randint(0, 100000)).zfill(5)

    return '{date}-{rand}'.format(date=datepart, rand=randpart)

class ModelController: 

    def __init__(self, model_name, flask_app): 

        correlation_id = cid()

        # Init variables
        self.model_name = model_name;
        self.ms_name = 'model-{}'.format(model_name)

        # Check if the model exists on the registry. 
        # If it does not, create it.
        self.model = check_registry(self.model_name, correlation_id)

        # Check if there's a champion model (pickle file) published on GCP Storage
        # If there's no model, upload the default model (local: erboh.v1)
        init_champion_model(self.model, correlation_id)

        # Event Consumers
        TotoEventConsumer(self.ms_name, ['erboh-predict-batch', 'erboh-predict-single', 'erboh-train'], [self.predict_batch, self.predict_single, self.train])

        # Event Publishers
        self.publisher_model_train = TotoEventPublisher(microservice=self.ms_name, topics=['erboh-train'])

        # APIs
        @flask_app.route('/')
        def smoke():
            return jsonify({ "api": self.ms_name, "status": "running" })

        @flask_app.route('/train', methods=['POST'])
        def train(): 

            # Start the training
            self.publisher_model_train.publish(topic='erboh-train', event={"correlationId": request.headers['x-correlation-id']})

            # Answer
            resp = jsonify({"message": "Training process started"})
            resp.status_code = 200
            resp.headers['Content-Type'] = 'application/json'

            return resp

        @flask_app.route('/score', methods=['GET'])
        def score(): 
            resp = jsonify(self.score(request))
            resp.status_code = 200
            resp.headers['Content-Type'] = 'application/json'

            return resp


    def train(self, request): 
        """
        Retrains the model 
        """
        TrainingProcess(self.model_name, request).do()

        return {"success": True, "message": "Model {} trained successfully".format(self.model_name)}

    def score(self, request): 
        """
        Calculate the accuracy (metrics) of the champion model
        """
        metrics = ScoreProcess(self.model, request).do()

        return {"metrics": metrics}

    def predict_single(self, message):
        """
        Predicts on a single item
        """
        predict_single(message, self.model)
    
    def predict_batch(self, message):
        """
        Predicts on a batch of items
        """
        predict_batch(message, self.model)


