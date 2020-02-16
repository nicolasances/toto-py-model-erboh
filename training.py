# Test of ML Flow

import re
from datetime import datetime as dt
from datetime import timedelta

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import classification_report, confusion_matrix, f1_score
#from sklearn.model_selection import train_test_split
#from sklearn.neural_network import MLPClassifier

features_red = pd.read_csv('features.csv')

X = features_red[['sacsw1_m1', 'sacsw2_m1', 'sacsw3m_m1', 'sac1_m1', 'sac2m_m1', 'sacd_m1', 'sacd3_m1',
            'sacsw1_m2', 'sacsw2_m2', 'sacsw3m_m2', 'sac1_m2', 'sac2m_m2', 'sacd_m2', 'sacd3_m2',
            'sacsw1_m3', 'sacsw2_m3', 'sacsw3m_m3', 'sac1_m3', 'sac2m_m3', 'sacd_m3', 'sacd3_m3',
            'sacsw1_m4', 'sacsw2_m4', 'sacsw3m_m4', 'sac1_m4', 'sac2m_m4', 'sacd_m4', 'sacd3_m4', 
            'sesm']]

y = features_red['occurs_monthly']

mlflow.set_tracking_uri('http://104.199.30.1:5000/')
mlflow.set_experiment('erboh2')

def train(i) : 
    # --------------------------------------
    # Training
    # --------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    with mlflow.start_run():

        best_nn = MLPClassifier(hidden_layer_sizes=(5, 5), activation='identity', alpha=0.1, max_iter=1000)

        best_nn.fit(X_train, y_train)

        best_nn_pred = best_nn.predict(X_test)

        f1 = f1_score(y_test, best_nn_pred)

        print("F1 score: {}".format(f1))  

        mlflow.log_param("alpha", 0.1)
        mlflow.log_param("activation", 'identity')
        mlflow.log_param("hidden_layer_sizes", (5, 5))
        mlflow.log_param('i', i)

        mlflow.log_metric("f1", f1)

        mlflow.sklearn.log_model(best_nn, "model")
        mlflow.log_artifact('data.labeled.nm.to202001.csv')
        mlflow.log_artifact('features.csv')

for i in range(10):
    print('Training #{}'.format(i))
    train(i)