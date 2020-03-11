import os
import uuid
import pandas as pd
from dlg.trainer_nogrid import Trainer
from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from dlg.predictor import Predictor
from dlg.score import Scorer
from remote.totoml_registry import post_retrained_model
from remote.gcpremote import save_retrained_model_pickle
from toto_logger.logger import TotoLogger

logger = TotoLogger()

class TrainingProcess: 

    def __init__(self, model_name, correltion_id): 
        self.user = 'all'
        self.correlation_id = correltion_id
        self.model_name = model_name
        self.context = 'TRAINING'

    def do(self) : 

        # Create the folder where to store all the data
        folder = "{tmp}/{model_name}/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], model_name=self.model_name, fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download all history
        history_filename = HistoryDownloader(folder, self.correlation_id, context=self.context).download(user=self.user)

        # 2. Engineer features
        # TRAINING = TRUE because we want to keep the "monthly" column 
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, self.correlation_id, training=True, context=self.context).do(user=self.user)

        # 3. Training
        (trained_model, train_features_filename, test_features_filename) = Trainer(folder, features_filename, model_feature_names, self.correlation_id, context=self.context).do()

        # 4. Predict and score
        (y_test_pred, y_test) = Predictor(test_features_filename, model_feature_names, self.correlation_id, predict_only_labeled=True, model=trained_model, context=self.context).do()

        metrics = Scorer(self.correlation_id, context=self.context).do(y_test, y_test_pred)

        # 5. Post the new model to TotoML Registry
        post_retrained_model(self.model_name, metrics, self.correlation_id, context=self.context)

        # 6. Save the pickle file to Storage
        save_retrained_model_pickle(self.model_name, trained_model, context=self.context)



