import os
import re
import uuid
import pandas as pd
from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from toto_logger.logger import TotoLogger

tmp_folder = os.environ['TOTO_TMP_FOLDER']
base_folder = "{tmp}/erboh".format(tmp=tmp_folder)
os.makedirs(name=base_folder, exist_ok=True)

logger = TotoLogger()

class Trainer: 

    def __init__(self, model_controller, request): 
        self.user = 'all'
        self.correlation_id = request.headers['x-correlation-id']
        self.model_controller = model_controller

    def do(self) : 

        # Create a UUID for the folder containing history, features and predictions
        fid = uuid.uuid1()
        # Create a folder
        folder = "{base_folder}/{fid}".format(base_folder=base_folder, fid=fid)
        os.mkdir(folder)

        # 1. Download all history
        # ATTENTION!! THE HISTORY IS DOWNLOADED FOR ALL USERS
        logger.compute(self.correlation_id, '[ STEP 1 - HISTORICAL ] - Starting historical data download', 'info')

        history_filename = '{folder}/history.{user}.csv'.format(user=self.user, folder=folder);

        history = HistoryDownloader(history_filename, self.correlation_id)
        history.download(self.user)

        if history.empty: 
            logger.compute(self.correlation_id, '[ STEP 1 - HISTORICAL ] - No historical data', 'warn')
            return

        logger.compute(self.correlation_id, '[ STEP 1 - HISTORICAL ] - Historical data downloaded', 'info')

        # 2. Build Features for historical data
        logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - Starting feature engineering', 'info')

        features_filename = '{folder}/features.{user}.csv'.format(user=self.user, folder=folder);
        
        feature_engineering = FeatureEngineering(history_filename, features_filename, training=True)
        feature_engineering.do()

        logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - Features engineered successfully', 'info')
        logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - # of rows: {}'.format(feature_engineering.count), 'info')

        if feature_engineering.empty: 
            logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - No rows to process. Stopping', 'warn')
            return {"inferedRows": feature_engineering.count}

        # 3. Training
        logger.compute(self.correlation_id, '[ STEP 3 - TRAINING ] - Starting training on historical data', 'info')

        try: 
            features = pd.read_csv(features_filename)
            
            # Only keep the features that are labeled!
            features = features[features['monthly'].notnull()]

            # Change the value of the monthly from bool to 0-1 values
            features['monthly'] = features['monthly'].apply(lambda x : int(x == True))

        except: 
            logger.compute(self.correlation_id, '[ STEP 3 - TRAINING ] - Problem reading file {}. Stopping'.format(features_filename), 'error')

        logger.compute(self.correlation_id, '[ STEP 3 - TRAINING ] - Training on {} rows'.format(len(features)),'info')

        X = features[feature_engineering.model_feature_names]
        y = features['monthly']

        best_nn = MLPClassifier(hidden_layer_sizes=(5, 5), activation='identity', alpha=0.1, max_iter=1000)

        # IMPORTANT
        # We're not splitting the data between test and train, because we're not doing
        # any tuning!! 
        # Right now we're just doing full retraining, with already tuned parameters!
        best_nn.fit(X, y)

        best_nn_pred = best_nn.predict(X)

        f1 = f1_score(y, best_nn_pred)

        logger.compute(self.correlation_id, '[ STEP 3 - TRAINING ] - Training complete. F1 Score: {}'.format(f1),'info')

        # 4. Save the new model as a challenger 
        self.model_controller.post_retrained_model({
            "model": best_nn, 
            "metrics": [
                {"name": "F1", "value": f1}
            ]
        }, self.correlation_id)

        logger.compute(self.correlation_id, '[ STEP 4 - POSTING ] - Posted new retrained model.','info')



