import datetime
from random import randint
from flask import Flask, jsonify, request, Response

from toto_pubsub.consumer import TotoEventConsumer
from toto_pubsub.publisher import TotoEventPublisher

from remote.totoml_registry import check_registry
from remote.gcpremote import init_champion_model, load_champion_model

def cid(): 
    '''
    Generates a Toto-valid Correlation ID
    Example: 20200229160021694-09776
    '''
    datepart = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    randpart = str(randint(0, 100000)).zfill(5)

    return '{date}-{rand}'.format(date=datepart, rand=randpart)

class ModelController: 
    """
    This class controls a model and acts as a proxy for all operations 
    regarding a model 
    """

    def __init__(self, model_name, flask_app, model_delegate): 
        """
        Initializes the controller

        Parameters
        ----------
        model_name (string)
            The name of the model (e.g. erboh)

        flask_app 
            The flask app
        
        model_delegate
            A model delegate, that provides the implementations of the following functions: 
            - predict_single()
            - predict_batch()
            - score()
            - train()
        """

        self.model_delegate = model_delegate

        # Generate a correlation ID for all init operations
        correlation_id = cid()

        # Init variables
        self.model_name = model_name;
        self.ms_name = 'model-{}'.format(model_name)

        # Check if the model exists on the registry. 
        # If it does not, create it.
        self.model_info = check_registry(self.model_name, correlation_id)

        # Check if there's a champion model (pickle file) published on GCP Storage
        # If there's no model, upload the default model (local: erboh.v1)
        init_champion_model(self.model_info, correlation_id)

        # Load the champion model in memory
        self.model = load_champion_model(self.model_info, correlation_id)

        # TODO : listen to a specific event for when a new model is 
        #        upgraded to champion so that you can reload 
        #        the champion model in memory

        # TODO : on startup, check if all the data is labeled, 
        #        otherwise use the champion model to label everything! 

        # TODO : support configuration of what should happen on a predict_single call:
        #        - support the possibility to not post the data to expenses but just get the prediction result

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
        self.model_delegate.train(self.model_info['name'], request)

        return {"success": True, "message": "Model {} trained successfully".format(self.model_info['name'])}

    def score(self, request): 
        """
        Calculate the accuracy (metrics) of the champion model
        """
        metrics = self.model_delegate.score(self.model_info, self.model, request.headers['x-correlation-id'])

        return {"metrics": metrics}

    def predict_single(self, message):
        """
        Predicts on a single item
        """
        self.model_delegate.predict_single(self.model, message)
    
    def predict_batch(self, message):
        """
        Predicts on a batch of items
        """
        self.model_delegate.predict_batch(self.model, message)


