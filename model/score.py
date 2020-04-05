import os
import uuid
import joblib
import pandas as pd

from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from dlg.predictor import Predictor
from dlg.score import Scorer

from totoml.model import ModelScore

from toto_logger.logger import TotoLogger

logger = TotoLogger()

class ScoreProcess: 

    def __init__(self): 
        self.user = 'all'
   
    def score(self, model, context): 
        """
        This method will compute the metrics of the model
        """
        correlation_id = context.correlation_id
        model_name = model.info['name']    
        context_process = context.process

        # Create the folder where to store all the data
        folder = "{tmp}/erboh/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download the historical data
        history_filename = HistoryDownloader(folder, correlation_id, context=context_process).download(user=self.user)

        # 2. Engineer features
        # TRAINING = TRUE because we want to keep the "monthly" column 
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, correlation_id, training=True, context=context_process).do(user=self.user)

        trained_model = joblib.load(model.files['model'])

        # 3. Predict on features
        (y_pred, y) = Predictor(features_filename, model_feature_names, correlation_id, predict_only_labeled=True, model=trained_model, context=context_process).do()

        # 4. Calculate accuracy
        score = Scorer(correlation_id, context=context_process).do(y, y_pred)

        return ModelScore(score, [history_filename, features_filename])

