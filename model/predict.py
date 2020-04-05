import joblib
import os
import uuid

from toto_logger.logger import TotoLogger

from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from dlg.predictor import Predictor

from totoml.model import ModelPrediction

from remote.expenses import update_expense

logger = TotoLogger()

class SinglePredictor: 

    def __init__(self): 
        pass

    def predict (self, model, context, data):
        '''
        Predicts the "monthly" classification of the provided expense
        '''
        correlation_id = context.correlation_id
        trained_model = joblib.load(model.files['model'])
        online = context.online
        context_process = context.process

        # Extract the data 
        try: 

            expense_id = data['id']
            user = data['user']
            category = data['category']
            amount = data['amount']
            description = data['description']
            date = data['date'] 

        except KeyError as ke: 
            logger.compute(correlation_id, "[ PREDICTION LISTENER ] - Event {} has attributes missing. Got error: {}".format(data, ke), 'error')

        # Create the folder where to store all the data
        folder = "{tmp}/erboh/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download the data for the 4 months before the expense date
        history_downloader = HistoryDownloader(folder, correlation_id, context=context_process)
        history_downloader.download_from(user=user, num_months=4, from_date=date)
        history_filename = history_downloader.append_expense(expense_id, user, amount, category, description, date)

        # 2. Feature Engineering
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, correlation_id, context=context_process).do(user=user)

        # 3. Load the model and predict
        (y_pred, y) = Predictor(features_filename, model_feature_names, correlation_id, model=trained_model, context=context_process).do()

        logger.compute(correlation_id, '[ {context} ] - [ PREDICT ] - Prediction: {p}'.format(context=context_process, p=y_pred[0]), 'info')

        # 4. Post an update to the expense
        if not online:
            update_expense({"id": expense_id, "monthly": y_pred[0]}, correlation_id, context=context_process)

            return ModelPrediction(files=[history_filename, features_filename])

        # Return the prediction
        return ModelPrediction(prediction={"expenseId": expense_id, "monthly": int(y_pred[0])}, files=[history_filename, features_filename])


# {"correlationId": "202002121919219199", "id": "5d71e5adcb15b1191e7ba273", "amount": 699.9, "user": "nicolas.matteazzi@gmail.com", "category": "AUTO", "description": "Train", "date": "20190906"}