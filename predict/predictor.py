import pandas as pd
import joblib
from toto_logger.logger import TotoLogger

logger = TotoLogger()

class Predictor: 

    def __init__(self, features_filename, predict_feature_names, cid, predict_only_labeled=False):
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
        """
        self.correlation_id = cid
        self.features_filename = features_filename
        self.predict_only_labeled = predict_only_labeled
        self.predict_feature_names = predict_feature_names
        self.model = joblib.load('erboh.v1')

    def do(self): 
        """
        Predicts based on the provided data
        """
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
            logger.compute(self.correlation_id, '[ PREDICTING ] - Problem reading file {}. Stopping'.format(self.features_filename), 'error')
            return

        logger.compute(self.correlation_id, '[ PREDICTING ] - Predicting on {} rows'.format(len(features)),'info')

        X = features[self.predict_feature_names]
        y = features['monthly']

        y_pred = self.model.predict(X)

        logger.compute(self.correlation_id, '[ PREDICTING ] - Prediction completed. Generated {} predictions'.format(len(y_pred)),'info')

        return (y_pred, y)
