import os
import uuid
import pandas as pd
from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from score.score import Scorer
from predict.predictor import Predictor
from toto_logger.logger import TotoLogger
from remote.totoml_registry import put_champion_metrics

logger = TotoLogger()

class ScoreProcess: 

    def __init__(self, model_name, request): 
        self.user = 'all'
        self.correlation_id = request.headers['x-correlation-id']
        self.model_name = model_name

    def do(self): 
        """
        This method will compute the metrics of the model
        and post them 
        """
        
        # Create the folder where to store all the data
        folder = "{tmp}/erboh/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download the historical data
        history_filename = HistoryDownloader(folder, self.correlation_id).download(user=self.user)

        # 2. Engineer features
        # TRAINING = TRUE because we want to keep the "monthly" column 
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, self.correlation_id, training=True).do(user=self.user)

        # 3. Predict on features
        (y_pred, y) = Predictor(features_filename, model_feature_names, self.correlation_id, predict_only_labeled=True).do()

        # 4. Calculate accuracy
        metrics = Scorer(self.correlation_id).do(y, y_pred)

        # 5. Post metrics
        put_champion_metrics(self.model_name, metrics, self.correlation_id)

        return metrics

