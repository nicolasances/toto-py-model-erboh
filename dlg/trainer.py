import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from toto_logger.logger import TotoLogger

logger = TotoLogger()

class Trainer: 

    def __init__(self, folder, features_filename, model_feature_names, cid):
        self.features_filename = features_filename
        self.correlation_id = cid
        self.model_feature_names = model_feature_names
        self.folder = folder

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

        # Train the model
        # Do a GridSearch CV for hyper param
        gs = GridSearchCV(MLPClassifier(activation='identity', max_iter=500), param_grid={
            'hidden_layer_sizes': [(5, 5), (9, 9)], 
            'alpha': [0.01, 0.1], 
        }, )

        gs.fit(X, y)
        
        results = gs.cv_results_

        print(results)

        logger.compute(self.correlation_id, '[ TRAINING ] - Model trained.','info')

        # Return the model and the split features files
        return (best_nn, features_filename, features_filename)
