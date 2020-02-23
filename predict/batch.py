import os
import requests
import joblib
import pandas as pd
import uuid

from toto_logger.logger import TotoLogger

from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from dlg.remote import ExpenseUpdater
from dlg.storage import FileStorage

tmp_folder = os.environ['TOTO_TMP_FOLDER']
if not os.path.exists(tmp_folder):
    os.mkdir(tmp_folder)
    
base_folder = "{tmp}/erboh".format(tmp=tmp_folder)

# Create the target folder if it does not exist
if not os.path.exists(base_folder):
    os.mkdir(base_folder)

model = joblib.load('erboh.v1')
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

        Predictor(user, cid).do()

    except KeyError as ke: 
        logger.compute(cid, "Event {} is missing attributes. Got error: {}".format(message, ke),'error')
    

class Predictor:

    def __init__(self, user, correlation_id):
        '''
        user: user email
        '''
        self.user = user
        self.correlation_id = correlation_id

    def do(self):

        # Create a UUID for the folder containing history, features and predictions
        fid = uuid.uuid1()
        # Create a folder
        folder = "{base_folder}/{fid}".format(base_folder=base_folder, fid=fid)
        os.mkdir(folder)

        # 1. Download all history
        logger.compute(self.correlation_id, '[ STEP 1 - HISTORICAL ] - Starting historical data download from API for {}'.format(self.user), 'info')

        history_filename = '{folder}/history.{user}.csv'.format(user=self.user, folder=folder);

        history = HistoryDownloader(history_filename, self.correlation_id)
        history.download(self.user)

        if history.empty: 
            logger.compute(self.correlation_id, '[ STEP 1 - HISTORICAL ] - No historical data for {}'.format(self.user), 'warn')
            return

        logger.compute(self.correlation_id, '[ STEP 1 - HISTORICAL ] - Historical data downloaded for {}'.format(self.user), 'info')

        # 2. Build Features for historical data
        logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - Starting feature engineering for {}'.format(self.user), 'info')

        features_filename = '{folder}/features.{user}.csv'.format(user=self.user, folder=folder);
        
        feature_engineering = FeatureEngineering(history_filename, features_filename)
        feature_engineering.do()

        logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - Features engineered successfully for {}'.format(self.user), 'info')
        logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - # of rows: {}'.format(feature_engineering.count), 'info')

        if feature_engineering.empty: 
            logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - No rows to process. Stopping', 'warn')
            return {"inferedRows": feature_engineering.count}

        # 3. Infer 
        logger.compute(self.correlation_id, '[ STEP 3 - INFERENCE ] - Starting inference on historical data for {}'.format(self.user), 'info')

        try: 
            features = pd.read_csv(features_filename)
        except: 
            logger.compute(self.correlation_id, '[ STEP 3 - INFERENCE ] - Problem reading file {}. Stopping'.format(features_filename), 'error')
            return {"inferedRows": feature_engineering.count}

        logger.compute(self.correlation_id, '[ STEP 3 - INFERENCE ] - Read {} rows from historical data'.format(len(features)),'info')

        # Extract the features into X
        X = features[feature_engineering.model_feature_names]

        # Run the model
        logger.compute(self.correlation_id, '[ STEP 3 - INFERENCE ] - Running bulk prediction', 'info')

        predictions = model.predict(X)

        # Create the output
        features['occurs_monthly'] = predictions

        logger.compute(self.correlation_id, '[ STEP 3 - INFERENCE ] - Predictions computed. Saving predictions to file.', 'info')

        # 4. Save Predictions
        predictions_filename = '{folder}/predictions.{user}.csv'.format(user=self.user, folder=folder)

        features.to_csv(predictions_filename)

        logger.compute(self.correlation_id, '[ STEP 3 - INFERENCE ] - Predictions generated and saved for {}'.format(self.user), 'info')

        # 5. For each prediction, update the expense (asynchronously)
        logger.compute(self.correlation_id, '[ STEP 4 - UPDATE ] - Updating payments with predictions', 'info')

        updater = ExpenseUpdater(self.correlation_id)
        updater.do(predictions_filename=predictions_filename)

        # 6. Save predictions to File Storage
        logger.compute(self.correlation_id, '[ STEP 5 - STORE ] - Store the prediction', 'info')

        file_storage.save_predictions(predictions_filename, self.user)

        logger.compute(self.correlation_id, '[ STEP 5 - STORE ] - Done!', 'info')

        return {"inferedRows": feature_engineering.count}

# Example: {"user": "nicolas.matteazzi@gmail.com", "correlationId": "9890182930123"}