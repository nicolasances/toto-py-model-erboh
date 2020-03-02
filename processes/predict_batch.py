import os
import requests
import joblib
import pandas as pd
import uuid

from toto_logger.logger import TotoLogger

from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from remote.gcpstorage import FileStorage
from remote.expenses import update_expenses
from predict.predictor import Predictor

logger = TotoLogger()
file_storage = FileStorage('model-erboh', 1)

def predict(message):
    '''
    Processes the batch inference request message
    '''
    cid = 'no-cid'
    try:
        cid = message['correlationId']
        user = message['user']

        logger.compute(cid, "[ BATCH INFER EVENT LISTENER ] - Received a request to do batch inference for {}".format(user), 'info')

        BatchPredictor(user, cid).do()

    except KeyError as ke: 
        logger.compute(cid, "Event {} is missing attributes. Got error: {}".format(message, ke),'error')
    

class BatchPredictor:

    def __init__(self, user, correlation_id):
        '''
        user: user email
        '''
        self.user = user
        self.correlation_id = correlation_id

    def do(self):

        # Create the folder where to store all the data
        folder = "{tmp}/erboh/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download all history
        history_filename = HistoryDownloader(folder, self.correlation_id).download(user=self.user)

        # 2. Build Features for historical data
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, self.correlation_id).do(user=self.user)

        # 3. Load the model and predict
        (y_pred, y, predictions_filename) = Predictor(features_filename, model_feature_names, self.correlation_id, save_to_folder=folder).do()

        # 5. For each prediction, update the expense (asynchronously)
        if y_pred is None: 
            return

        update_expenses(predictions_filename, self.correlation_id)

        # 6. Save predictions to File Storage & recalc accuracy
        # logger.compute(self.correlation_id, '[ STEP 5 - STORE ] - Store the prediction', 'info')

        # file_storage.save_predictions_and_accuracy(predictions_filename, self.user)

        # logger.compute(self.correlation_id, '[ STEP 5 - STORE ] - Done!', 'info')

        # return {"inferedRows": feature_engineering.count}

# Example: {"user": "nicolas.matteazzi@gmail.com", "correlationId": "test-predict-batch"}