import os
import requests
import joblib
import pandas as pd
import uuid

from totoml.model import ModelPrediction

from toto_logger.logger import TotoLogger

from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from dlg.predictor import Predictor

from remote.expenses import update_expenses

logger = TotoLogger()

class BatchPredictor:

    def __init__(self):
        pass

    def predict (self, model, context, data):

        correlation_id = context.correlation_id
        trained_model = joblib.load(model.files['model'])
        context_process = context.process

        user = 'all'
        if data is not None and "user" in data: 
            user = data['user']

        # Create the folder where to store all the data
        folder = "{tmp}/erboh/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download all history
        history_filename = HistoryDownloader(folder, correlation_id, context=context_process).download(user=user)

        # 2. Build Features for historical data
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, correlation_id, context=context_process).do(user=user)

        # 3. Load the model and predict
        (y_pred, y, predictions_filename) = Predictor(features_filename, model_feature_names, correlation_id, save_to_folder=folder, model=trained_model, context=context_process).do()

        # 5. For each prediction, update the expense (asynchronously)
        if y_pred is None: 
            return ModelPrediction(files=[history_filename, features_filename])

        update_expenses(predictions_filename, correlation_id, context=context_process)

        return ModelPrediction(files=[history_filename, features_filename])

# Example: {"user": "nicolas.matteazzi@gmail.com", "correlationId": "test-predict-batch"}