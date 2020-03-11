import pandas as pd
import numpy as np
from toto_logger.logger import TotoLogger
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

logger = TotoLogger()

class Trainer: 

    def __init__(self, folder, features_filename, model_feature_names, cid, context=''):
        self.features_filename = features_filename
        self.correlation_id = cid
        self.model_feature_names = model_feature_names
        self.folder = folder
        self.context = context

    def do(self): 
        
        logger.compute(self.correlation_id, '[ {context} ] - [ TRAINING ] - Starting training on historical data'.format(context=self.context), 'info')

        try: 
            features = pd.read_csv(self.features_filename)
            
            # Only keep the features that are labeled!
            features = features[features['monthly'].notnull()]

            # Change the value of the monthly from bool to 0-1 values
            features['monthly'] = features['monthly'].apply(lambda x : int(x == True))

        except: 
            logger.compute(self.correlation_id, '[ {context} ] - [ TRAINING ] - Problem reading file {f}. Stopping'.format(context=self.context, f=self.features_filename), 'error')
            return

        logger.compute(self.correlation_id, '[ {context} ] - [ TRAINING ] - Training on {r} rows'.format(context=self.context, r=len(features)),'info')

        X = features[self.model_feature_names]
        y = features['monthly']

        # Split train and test set, cause the accuracy is going to be calculated on the test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=100)

        # Save the sets 
        train_filename = '{folder}/features_train.csv'.format(folder=self.folder);
        test_filename = '{folder}/features_test.csv'.format(folder=self.folder);

        train_df = pd.DataFrame(X_train, columns=self.model_feature_names)
        test_df = pd.DataFrame(X_test, columns=self.model_feature_names)
        train_df['monthly'] = y_train
        test_df['monthly'] = y_test

        train_df.to_csv(train_filename)
        test_df.to_csv(test_filename)

        # Train the model
        best_nn = MLPClassifier(hidden_layer_sizes=(5, 5), activation='identity', alpha=0.1, max_iter=1000)
        best_nn.fit(X_train, y_train)

        logger.compute(self.correlation_id, '[ {context} ] - [ TRAINING ] - Model trained.'.format(context=self.context),'info')

        # Return the model and the split features files
        return (best_nn, train_filename, test_filename)
