import os
import uuid
import datetime 
import dateutil.relativedelta
import pandas as pd
import numpy as np
import joblib
from random import randint

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
file_storage = FileStorage('model-erboh', 1)
logger = TotoLogger()

def predict(message): 
    '''
    Processes a single prediction
    
    Requires a message that is formatted like this: 
    id (string): the id of the expense
    user (string): the user email
    amount (float): the amount of the expense (always positive)
    description (string): the description of the expense
    date (string): the date of the expense, formatted YYYYMMDD
    category (string): the category of the epense
    '''
    try: 

        expense_id = message['id']
        user = message['user']
        category = message['category']
        amount = message['amount']
        description = message['description']
        date = message['date'] 
        cid = message['correlationId']

        logger.compute(cid, "[ PREDICTION LISTENER ] - Received a request for a prediction on expense (id: {expense_id}, user: {user}, amount: {amount}, category: {category}, description: {description}, date: {date})".format(user=user, amount=amount, description=description, date=date, category=category, expense_id=expense_id), 'info')

        Predictor(cid).predict(expense_id, user, amount, category, description, date)
    
    except KeyError as ke: 
        print("Event {} has attributes missing. Got error: {}".format(message, ke))


class Predictor: 

    def __init__(self, correlation_id): 
        self.correlation_id = correlation_id

    def predict(self, expense_id, user, amount, category, description, date): 
        '''
        Predicts the "monthly" classification of the provided expense
        '''

        # Create a UUID for the folder containing history, features and predictions
        fid = uuid.uuid1()
        # Create a folder
        folder = "{base_folder}/{fid}".format(base_folder=base_folder, fid=fid)
        os.mkdir(folder)

        random_num = randint(0, 10000000)

        # 1. Download the last 4 months - those are used for the feature engineering
        logger.compute(self.correlation_id, '[ STEP 1 - HISTORICAL ] - Starting historical data download from API for {}'.format(user), 'info')
        # Extract the expense date
        date_obj = datetime.datetime.strptime(date, '%Y%m%d')
        ym = str(date_obj.year) + str(date_obj.month).rjust(2, '0')

        # Find the right month
        date_obj = date_obj.replace(day=1)
        date_obj -= dateutil.relativedelta.relativedelta(months=4)
        
        dateGte = date_obj.strftime('%Y%m%d')

        # Call the history downloader
        history_filename = "{folder}/history.{user}.{random_num}.csv".format(user=user, random_num=random_num, folder=folder)

        history = HistoryDownloader(history_filename, self.correlation_id)
        history.download(user, dateGte=dateGte)

        if history.empty: 
            logger.compute(self.correlation_id, '[ STEP 1 - HISTORY DOWNLOAD ] - No data retrieved from history', 'warn')
            return

        # 2. Do Feature Engineering
        logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - Starting feature engineering for {}'.format(user), 'info')
        
        # Append the new expense to the history file, so that we can reuse the feature engineering the way it is
        history_df = pd.read_csv(history_filename)
        new_df = pd.DataFrame(np.array([[expense_id, amount, category, date, description, None, ym]]), columns=['id','amount','category','date','description','monthly','yearMonth'])

        full_df = pd.concat([history_df, new_df], axis=0, ignore_index=True, sort=False)

        # Save to a new file
        history_ext_filename = history_filename[:-4] + '.ext.csv'

        full_df.to_csv(history_ext_filename)

        # Do Feature Engineering
        features_filename = '{folder}/features.{user}.{random_num}.csv'.format(user=user, random_num=random_num, folder=folder);
        
        feature_engineering = FeatureEngineering(history_ext_filename, features_filename)
        feature_engineering.do()

        features_df = pd.read_csv(features_filename)

        if feature_engineering.empty: 
            logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - No rows to process. Stopping', 'warn')
            return {"inferedRows": feature_engineering.count}

        logger.compute(self.correlation_id, '[ STEP 2 - FEATURE ENGINEERING ] - Features engineered successfully for {}'.format(user), 'info')

        # 3. Load the model and predict
        logger.compute(self.correlation_id, '[ STEP 3 - PREDICT ] - Starting inference on historical data for {}'.format(user), 'info')

        try: 
            features = pd.read_csv(features_filename)
        except: 
            logger.compute(self.correlation_id, '[ STEP 3 - PREDICT ] - Problem reading file {}. Stopping'.format(features_filename), 'info')
            return {"inferedRows": feature_engineering.count}

        logger.compute(self.correlation_id, '[ STEP 3 - PREDICT ] - Read {} rows from historical data'.format(len(features)), 'info')

        # Extract the features into X
        X = features[feature_engineering.model_feature_names]

        # Run the model
        logger.compute(self.correlation_id, '[ STEP 3 - PREDICT ] - Running prediction', 'info')

        predictions = model.predict(X)

        prediction = predictions[0]

        logger.compute(self.correlation_id, '[ STEP 3 - PREDICT ] - Prediction: {}'.format(prediction), 'info')

        # 4. Post an update to the expense
        logger.compute(self.correlation_id, '[ STEP 4 - UPDATE ] - Updating payment with prediction', 'info')

        expense = {
            "id": expense_id, 
            "monthly": prediction
        }
        
        ExpenseUpdater(self.correlation_id).do(expense=expense)

        # 5. Update the predictions file
        logger.compute(self.correlation_id, '[ STEP 5 - STORE ] - Store the prediction', 'info')

        file_storage.save_prediction(prediction=expense, user=user)

        logger.compute(self.correlation_id, '[ STEP 5 - STORE ] - Done!', 'info')


#{"correlationId": "202002121919219199", "id": "5d71e5adcb15b1191e7ba273", "amount": 699.9, "user": "nicolas.matteazzi@gmail.com", "category": "AUTO", "description": "Train", "date": "20190906"}