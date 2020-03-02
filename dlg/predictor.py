import pandas as pd
import joblib
from toto_logger.logger import TotoLogger
from remote.gcpremote import load_champion_model

logger = TotoLogger()

class Predictor: 

    def __init__(self, features_filename, predict_feature_names, cid, predict_only_labeled=False, model=None, save_to_folder=None):
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

        model (dict) default None
            The model to use for the prediction. 
            Dictionnary as returned by the Toto ML Registry GET /models/:name
            If None is passed, the default local model is going to be used
            The Dict must have: 
            - name (string): name of the model to load
            - version (int): version of the model to load
        """
        self.correlation_id = cid
        self.features_filename = features_filename
        self.predict_only_labeled = predict_only_labeled
        self.predict_feature_names = predict_feature_names
        self.save_to_folder = save_to_folder

        if model is not None: 
            # Load the model
            logger.compute(self.correlation_id, '[ PREDICTING ] - Loading model {model}.v{version} from storage for prediction.'.format(model=model['name'], version=model['version']),'info')

            self.model = load_champion_model(model['name'], model['version'], self.correlation_id)
        else:
            logger.compute(self.correlation_id, '[ PREDICTING ] - No model passed. Using the local default.','info')

            self.model = joblib.load('erboh.v1')

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
            logger.compute(self.correlation_id, '[ PREDICTING ] - Problem reading file {}. Stopping'.format(self.features_filename), 'error')
            return (None, None, None)

        logger.compute(self.correlation_id, '[ PREDICTING ] - Predicting on {} rows'.format(len(features)),'info')

        X = features[self.predict_feature_names]

        y = None
        if 'monthly' in features.columns:
            y = features['monthly']

        y_pred = self.model.predict(X)

        logger.compute(self.correlation_id, '[ PREDICTING ] - Prediction completed. Generated {} predictions'.format(len(y_pred)),'info')

        if self.save_to_folder != None:
            # Save to file
            predictions_filename = '{folder}/predictions.csv'.format(folder=self.save_to_folder)

            features['occurs_monthly'] = y_pred
            features.to_csv(predictions_filename)

            logger.compute(self.correlation_id, '[ PREDICTING ] - Predictions saved on disk: {}'.format(predictions_filename), 'info')

            return (y_pred, y, predictions_filename)

        return (y_pred, y)
