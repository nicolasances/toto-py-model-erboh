import os
import uuid
import joblib
import pandas as pd

from dlg.trainer_nogrid import Trainer
from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from dlg.predictor import Predictor
from dlg.score import Scorer

from toto_logger.logger import TotoLogger

from totoml.model import TrainedModel

logger = TotoLogger()

class TrainingProcess: 

    def __init__(self): 
        self.user = 'all'

    def train(self, model_info, context):

        model_name = model_info['name']
        correlation_id = context.correlation_id
        context_process = context.process

        # Create the folder where to store all the data
        folder = "{tmp}/{model_name}/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], model_name=model_name, fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download all history
        history_filename = HistoryDownloader(folder, correlation_id, context=context_process).download(user=self.user)

        # 2. Engineer features
        # TRAINING = TRUE because we want to keep the "monthly" column 
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, correlation_id, training=True, context=context_process).do(user=self.user)

        # 3. Training
        (trained_model, train_features_filename, test_features_filename) = Trainer(folder, features_filename, model_feature_names, correlation_id, context=context_process).do()

        # 4. Predict and score
        (y_test_pred, y_test) = Predictor(test_features_filename, model_feature_names, correlation_id, predict_only_labeled=True, model=trained_model, context=context_process).do()

        score = Scorer(correlation_id, context=context_process).do(y_test, y_test_pred)

        # 5. Save all the objects
        model_filepath = "{folder}/model".format(folder=folder)

        joblib.dump(trained_model, model_filepath)

        return TrainedModel({"model": model_filepath}, [history_filename, features_filename], score)



