import datetime
from random import randint
from flask import Flask, jsonify, request, Response
from apscheduler.schedulers.background import BackgroundScheduler

from toto_pubsub.consumer import TotoEventConsumer
from toto_pubsub.publisher import TotoEventPublisher
from toto_logger.logger import TotoLogger

from remote.totoml_registry import check_registry as align_registry
from remote.gcpremote import init_champion_model, load_champion_model

logger = TotoLogger()

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

    def __init__(self, model_name, flask_app, model_delegate, config=None): 
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

        config (dict)
            A configuration object 
            - train_cron (dict), optional a dictionnary with the following fields: hour, minute, second that 
            set the periodicity for the training process
        """

        self.model_delegate = model_delegate

        # Generate a correlation ID for all init operations
        correlation_id = cid()

        # Init variables
        self.model_name = model_name;
        self.ms_name = 'model-{}'.format(model_name)

        # Check if the model exists on the registry. 
        # If it does not, create it.
        self.model_info = align_registry(self.model_name, correlation_id)

        # Check if there's a champion model (pickle file) published on GCP Storage
        # If there's no model, upload the default model (local)
        init_champion_model(self.model_info, correlation_id)

        # Load the champion model in memory
        self.model = load_champion_model(self.model_info, correlation_id)

        # TODO : listen to a specific event for when a new model is 
        #        upgraded to champion so that you can reload 
        #        the champion model in memory

        # Event Consumers
        TotoEventConsumer(self.ms_name, ['{model}-predict-batch'.format(model=self.model_info['name']), '{model}-predict-single'.format(model=self.model_info['name']), '{model}-train'.format(model=self.model_info['name'])], [self.predict_batch, self.predict_single, self.train])

        # Event Publishers
        self.publisher_model_train = TotoEventPublisher(microservice=self.ms_name, topics=['{model}-train'.format(model=self.model_info['name'])])

        # APIs
        @flask_app.route('/')
        def smoke():
            return jsonify({ "api": self.ms_name, "status": "running" })

        @flask_app.route('/train', methods=['POST'])
        def train(): 

            # Topic to which the train message will be pushed
            topic = '{model}-train'.format(model=self.model_info['name'])
            event = {"correlationId": request.headers['x-correlation-id']}

            # Start the training
            self.publisher_model_train.publish(topic=topic, event=event)

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

        @flask_app.route('/predict', methods=['POST'])
        def predict(): 

            data = request.json
            correlation_id = request.headers['x-correlation-id']

            resp = jsonify(self.predict_single(data, correlation_id, online=True))
            resp.status_code = 200
            resp.headers['Content-Type'] = 'application/json'

            return resp


    def train(self, request=None): 
        """
        Retrains the model 
        """
        correlation_id = cid()

        if request is not None and 'correlationId' in request: 
            correlation_id = request['correlationId']

        self.model_delegate.train(self.model_info['name'], correlation_id)

        return {"success": True, "message": "Model {} trained successfully".format(self.model_info['name'])}

    def score(self, request=None): 
        """
        Calculate the accuracy (metrics) of the champion model
        """
        correlation_id = cid()

        if request is not None and 'x-correlation-id' in request.headers: 
            correlation_id = request.headers['x-correlation-id']
        
        metrics = self.model_delegate.score(self.model_info, self.model, correlation_id)

        return {"metrics": metrics}

    def predict_single(self, data, correlation_id=None, online=False):
        """
        Predicts on a single item
        """
        if correlation_id is None and 'correlationId' in data: 
            correlation_id = data['correlationId']
        else: 
            correlation_id = cid()

        return self.model_delegate.predict_single(self.model, data, correlation_id, online)
    
    def predict_batch(self, data=None):
        """
        Predicts on a batch of items
        """
        if data is not None and 'correlationId' in data: 
            correlation_id = data['correlationId']
        else: 
            correlation_id = cid()

        self.model_delegate.predict_batch(self.model, correlation_id, data=data)


