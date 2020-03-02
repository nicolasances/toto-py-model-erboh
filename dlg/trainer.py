import pandas as pd
from toto_logger.logger import TotoLogger
from sklearn.neural_network import MLPClassifier

logger = TotoLogger()

class Trainer: 

    def __init__(self, features_filename, model_feature_names, cid):
        self.features_filename = features_filename
        self.correlation_id = cid
        self.model_feature_names = model_feature_names

    def do(self): 
        
        logger.compute(self.correlation_id, '[ TRAINING ] - Starting training on historical data', 'info')

        try: 
            features = pd.read_csv(self.features_filename)
            
            # Only keep the features that are labeled!
            features = features[features['monthly'].notnull()]

            # Change the value of the monthly from bool to 0-1 values
            features['monthly'] = features['monthly'].apply(lambda x : int(x == True))

        except: 
            logger.compute(self.correlation_id, '[ TRAINING ] - Problem reading file {}. Stopping'.format(self.features_filename), 'error')
            return

        logger.compute(self.correlation_id, '[ TRAINING ] - Training on {} rows'.format(len(features)),'info')

        X = features[self.model_feature_names]
        y = features['monthly']

        best_nn = MLPClassifier(hidden_layer_sizes=(5, 5), activation='identity', alpha=0.1, max_iter=1000)

        # IMPORTANT
        # We're not splitting the data between test and train, because we're not doing
        # any tuning!! 
        # Right now we're just doing full retraining, with already tuned parameters!
        best_nn.fit(X, y)

        logger.compute(self.correlation_id, '[ TRAINING ] - Model trained.','info')

        return best_nn
