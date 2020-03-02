import os
import uuid
import pandas as pd
from dlg.trainer import Trainer
from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from dlg.predictor import Predictor
from dlg.score import Scorer
from remote.totoml_registry import post_retrained_model
from remote.gcpremote import save_retrained_model_pickle
from toto_logger.logger import TotoLogger

logger = TotoLogger()

class TrainingProcess: 

    def __init__(self, model_name, message): 
        self.user = 'all'
        self.correlation_id = message['correlationId']
        self.model_name = model_name

    def do(self) : 

        # Create the folder where to store all the data
        folder = "{tmp}/{model_name}/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], model_name=self.model_name, fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download all history
        history_filename = HistoryDownloader(folder, self.correlation_id).download(user=self.user)

        # 2. Engineer features
        # TRAINING = TRUE because we want to keep the "monthly" column 
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, self.correlation_id, training=True).do(user=self.user)

        # 3. Training
        trained_model = Trainer(features_filename, model_feature_names, self.correlation_id).do()

        # 4. Predict and score
        (y_pred, y) = Predictor(features_filename, model_feature_names, self.correlation_id, predict_only_labeled=True, model=trained_model).do()

        metrics = Scorer(self.correlation_id).do(y, y_pred)

        # 5. Post the new model to TotoML Registry
        post_retrained_model(self.model_name, metrics, self.correlation_id)

        # 6. Save the pickle file to Storage
        save_retrained_model_pickle(self.model_name, trained_model)



