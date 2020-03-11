import os
import requests
import joblib
import pandas as pd
import uuid

from toto_logger.logger import TotoLogger

from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from dlg.predictor import Predictor
from remote.gcpstorage import FileStorage
from remote.expenses import update_expenses

logger = TotoLogger()

class BatchPredictor:

    def __init__(self, model, correlation_id, user=None):
        '''
        user: user email
        '''
        self.user = user
        self.correlation_id = correlation_id
        self.model = model
        self.context = 'PREDICTION (BATCH)'

        if self.user == None:
            self.user = 'all'

    def do(self):

        # Create the folder where to store all the data
        folder = "{tmp}/erboh/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download all history
        history_filename = HistoryDownloader(folder, self.correlation_id, context=self.context).download(user=self.user)

        # 2. Build Features for historical data
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, self.correlation_id, context=self.context).do(user=self.user)

        # 3. Load the model and predict
        (y_pred, y, predictions_filename) = Predictor(features_filename, model_feature_names, self.correlation_id, save_to_folder=folder, model=self.model, context=self.context).do()

        # 5. For each prediction, update the expense (asynchronously)
        if y_pred is None: 
            return

        update_expenses(predictions_filename, self.correlation_id, context=self.context)

# Example: {"user": "nicolas.matteazzi@gmail.com", "correlationId": "test-predict-batch"}