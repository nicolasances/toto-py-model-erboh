import pandas as pd
import joblib
from toto_logger.logger import TotoLogger
from remote.gcpremote import load_champion_model

logger = TotoLogger()

class Predictor: 

    def __init__(self, features_filename, predict_feature_names, cid, model, predict_only_labeled=False, save_to_folder=None, context=''):
        """
        Constructor

        Parameters
        ----------
        predict_only_labeled (boolean) default False
            Specifies whether the prediction should only be done on data that has labels
            This is for example used when wanting to use the predict to calculate accuracy 
            (which means that the labels have to be there for the comparison) or in the training process
            (also to calculate the accuracy)

        predict_feature_names (list) 
            The list of the features that HAVE TO BE CONSIDERED FOR PREDICTING

        model (object) MANDATORY
            The model pickle file to use for the prediction. 
        """
        self.correlation_id = cid
        self.features_filename = features_filename
        self.predict_only_labeled = predict_only_labeled
        self.predict_feature_names = predict_feature_names
        self.save_to_folder = save_to_folder
        self.model = model
        self.context = context

    def do(self): 
        """
        Predicts based on the provided data
        """

        if self.features_filename == None: 
            return (None, None, None)

        try: 
            # 1. Get the features
            features = pd.read_csv(self.features_filename)
            
            if self.predict_only_labeled:
                # Only keep the features that are labeled!
                features = features[features['monthly'].notnull()]

            # Change the value of the monthly from bool to 0-1 values
            if 'monthly' in features.columns:
                features['monthly'] = features['monthly'].apply(lambda x : int(x == True))

        except:
            logger.compute(self.correlation_id, '[ {context} ] - [ PREDICTING ] - Problem reading file {f}. Stopping'.format(context=self.context, f=self.features_filename), 'error')
            return (None, None, None)

        logger.compute(self.correlation_id, '[ {context} ] - [ PREDICTING ] - Predicting on {r} rows'.format(context=self.context, r=len(features)),'info')

        X = features[self.predict_feature_names]

        y = None
        if 'monthly' in features.columns:
            y = features['monthly']

        y_pred = self.model.predict(X)

        logger.compute(self.correlation_id, '[ {context} ] - [ PREDICTING ] - Prediction completed. Generated {p} predictions'.format(context=self.context, p=len(y_pred)),'info')

        if self.save_to_folder != None:
            # Save to file
            predictions_filename = '{folder}/predictions.csv'.format(folder=self.save_to_folder)

            features['occurs_monthly'] = y_pred
            features.to_csv(predictions_filename)

            logger.compute(self.correlation_id, '[ {context} ] - [ PREDICTING ] - Predictions saved on disk: {f}'.format(context=self.context, f=predictions_filename), 'info')

            return (y_pred, y, predictions_filename)

        return (y_pred, y)
