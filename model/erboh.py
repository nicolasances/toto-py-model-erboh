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

    def predict_single(self, model, data, correlation_id, online=False): 
        
        try: 

            expense_id = data['id']
            user = data['user']
            category = data['category']
            amount = data['amount']
            description = data['description']
            date = data['date'] 

            return SinglePredictor(model, correlation_id, online).predict(expense_id, user, amount, category, description, date)
        
        except KeyError as ke: 
            logger.compute(correlation_id, "[ PREDICTION LISTENER ] - Event {} has attributes missing. Got error: {}".format(data, ke), 'error')

    def predict_batch(self, model, correlation_id, data=None): 

        user = 'all'
        if data is not None and "user" in data: 
            user = data['user']

        BatchPredictor(model, correlation_id, user).do()

    def train(self, model_name, correlation_id): 

        TrainingProcess(model_name, correlation_id).do()

    def score(self, model_info, model, cid):

        return ScoreProcess(model_info, model, cid).do()

        