import os
import uuid

from toto_logger.logger import TotoLogger

from processes.predict_single import SinglePredictor
from processes.predict_batch import BatchPredictor
from processes.training import TrainingProcess
from processes.scoring import ScoreProcess

logger = TotoLogger()

class ModelDelegate: 

    def predict_single(self, model, request): 

        try: 

            expense_id = request['id']
            user = request['user']
            category = request['category']
            amount = request['amount']
            description = request['description']
            date = request['date'] 
            cid = request['correlationId']

            SinglePredictor(model, cid).predict(expense_id, user, amount, category, description, date)
        
        except KeyError as ke: 
            logger.compute(cid, "[ PREDICTION LISTENER ] - Event {} has attributes missing. Got error: {}".format(request, ke), 'error')

    def predict_batch(self, model, request): 

        try:
            cid = request['correlationId']
            user = request['user']

            BatchPredictor(user, model, cid).do()

        except KeyError as ke: 
            logger.compute(cid, "[ BATCH INFER ] - Event {} is missing attributes. Got error: {}".format(request, ke),'error')

    def train(self, model_name, request): 

        TrainingProcess(model_name, request).do()

    def score(self, model_info, model, cid):

        return ScoreProcess(model_info, model, cid).do()

        