import os
import uuid
import datetime
from random import randint

from toto_logger.logger import TotoLogger

from processes.predict_single import SinglePredictor
from processes.predict_batch import BatchPredictor
from processes.training import TrainingProcess
from processes.scoring import ScoreProcess

logger = TotoLogger()

def cid(): 
    '''
    Generates a Toto-valid Correlation ID
    Example: 20200229160021694-09776
    '''
    datepart = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    randpart = str(randint(0, 100000)).zfill(5)

    return '{date}-{rand}'.format(date=datepart, rand=randpart)

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

    def predict_batch(self, model, request=None): 

        corrid = cid()
        user = None

        if request is not None: 
            if 'correlationId' in request: 
                corrid = request['correlationId']
            if user in request: 
                user = request['user']

        BatchPredictor(model, corrid, user).do()

    def train(self, model_name, request): 

        TrainingProcess(model_name, request).do()

    def score(self, model_info, model, cid):

        return ScoreProcess(model_info, model, cid).do()

        