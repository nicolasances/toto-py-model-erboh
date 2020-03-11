import os
import uuid

from toto_logger.logger import TotoLogger

from dlg.history import HistoryDownloader
from dlg.feature import FeatureEngineering
from dlg.predictor import Predictor
from remote.expenses import update_expense
from remote.gcpstorage import FileStorage

logger = TotoLogger()

class SinglePredictor: 

    def __init__(self, model, correlation_id, online): 
        self.correlation_id = correlation_id
        self.model = model
        self.online = online
        self.context = 'PREDICTION (SINGLE)'

    def predict(self, expense_id, user, amount, category, description, date): 
        '''
        Predicts the "monthly" classification of the provided expense
        '''
        # Create the folder where to store all the data
        folder = "{tmp}/erboh/{fid}".format(tmp=os.environ['TOTO_TMP_FOLDER'], fid=uuid.uuid1())
        os.makedirs(name=folder, exist_ok=True)

        # 1. Download the data for the 4 months before the expense date
        history_downloader = HistoryDownloader(folder, self.correlation_id, context=self.context)
        history_downloader.download_from(user=user, num_months=4, from_date=date)
        history_filename = history_downloader.append_expense(expense_id, user, amount, category, description, date)

        # 2. Feature Engineering
        (model_feature_names, features_filename) = FeatureEngineering(folder, history_filename, self.correlation_id, context=self.context).do(user=user)

        # 3. Load the model and predict
        (y_pred, y) = Predictor(features_filename, model_feature_names, self.correlation_id, model=self.model, context=self.context).do()

        logger.compute(self.correlation_id, '[ {context} ] - [ PREDICT ] - Prediction: {p}'.format(context=self.context, p=y_pred[0]), 'info')

        # 4. Post an update to the expense
        if not self.online:
            update_expense({"id": expense_id, "monthly": y_pred[0], context=self.context}, self.correlation_id)

        # Return the prediction
        return {"expenseId": expense_id, "monthly": int(y_pred[0])}


# {"correlationId": "202002121919219199", "id": "5d71e5adcb15b1191e7ba273", "amount": 699.9, "user": "nicolas.matteazzi@gmail.com", "category": "AUTO", "description": "Train", "date": "20190906"}