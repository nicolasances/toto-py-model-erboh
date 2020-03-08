import os
import pandas as pd
import numpy as np
import uuid
from google.cloud import storage
from google.api_core.exceptions import NotFound
from toto_logger.logger import TotoLogger
from sklearn.metrics import precision_recall_fscore_support

client = storage.Client()

toto_env = os.environ['TOTO_ENV']

logger = TotoLogger()

class FileStorage: 
    '''
    This class gives access to the file storage for the model
    '''

    def __init__(self, model_name, model_version):
        '''
        Initializes the FileStorage object

        Parameters: 
        - model_name (string): the name of the model
        - model_version (int): the version of the model
        '''
        self.model_name = model_name
        self.model_version = model_version
        self.bucket_name = 'toto-{env}-model-storage'.format(env=toto_env)

        try:
            self.bucket = client.get_bucket(self.bucket_name)
        except NotFound: 
            logger.compute('no-id', 'Bucket {} not found. Please create it first'.format(self.bucket_name), 'error')
            return

    def create_tmp_filename(self): 
        ''' 
        This function just creates a temporary filename
        '''
        # Generate a filename
        filename = uuid.uuid1()

        # Define the target folder, create if it does not exist
        folder = os.environ['TOTO_TMP_FOLDER']
        os.makedirs(name=folder, exist_ok=True)

        return '{folder}/{file}'.format(folder=folder, file=filename)

    def get_bucket_object(self, user): 
        '''
        Returns the blob object corresponding to the stored predictions file 
        for the specified user
        '''
        pred_obj_name = '{model}/{version}/predictions/{user}.predictions.csv'.format(model=self.model_name, version=self.model_version, user=user)
        pred_obj = self.bucket.blob(pred_obj_name)

        return pred_obj

    def get_accuracy_bucket_obj(self, user): 
        '''
        Returns the blob object corresponding to the file where model
        accuracy info is saved
        '''
        acc_obj_name = '{model}/{version}/accuracy/{user}.accuracy.csv'.format(model=self.model_name, version=self.model_version, user=user)
        acc_obj = self.bucket.blob(acc_obj_name)

        return acc_obj

    def calc_and_save_accuracy(self, data, user):
        '''
        Calculates and saves the accuracy for the passed dataset. 
        Requires the dataset to provide the following columns: "prediction", "actual"
        '''
        # Check and correct data types
        if (data['actual'].dtype != np.float64): 
            data['actual'] = pd.to_numeric(data['actual'])
        if (data['prediction'].dtype != np.float64):
            data['prediction'] = pd.to_numeric(data['prediction'])

        # Calculate accuracy
        accuracy = precision_recall_fscore_support(data['actual'], data['prediction'])

        # Prepare data frame and save
        # This model is mostly interested in :
        # - Recall on class 1: did I manage to catch all 1s (all actual monthly expenses)
        # – Precision on class 1: did I manage to not wrongly classify stuff as "monthly"
        acc_df = pd.DataFrame(accuracy, columns=['Class 0', 'Class 1'], index=['Precision', 'Recall', 'F1', 'Support'])

        # Save the data
        tmp_acc_filename = self.create_tmp_filename()

        acc_df.to_csv(tmp_acc_filename)

        # Upload the data
        acc_obj = self.get_accuracy_bucket_obj(user)

        acc_obj.upload_from_filename(tmp_acc_filename)

        # Delete the tmp file
        os.remove(tmp_acc_filename)

        return acc_df

    def save(self, new_predictions, user):
        '''
        Method that groups the common logic for both save_prediction() and save_predictions()

        Requires new_predictions to be a DataFrame with the following columns: "id", "prediction"

        This also recalculate the accuracy (since the data is all in memory)
        '''
        # Load previous predictions from bucket, if any
        pred_obj = self.get_bucket_object(user)

        # If there are no previous predictions, just save all the predictions
        if not pred_obj.exists():
            # Add a column "actual" for the actual value and initialize to the value of the prediction
            new_predictions['actual'] = new_predictions['prediction']

            # Create a tmp local file and upload to Storage
            tmp_filename = self.create_tmp_filename()
            
            new_predictions.to_csv(tmp_filename)

            pred_obj.upload_from_filename(tmp_filename)

            # Delete the tmp file
            os.remove(tmp_filename)

            # Calculate and save accuracy
            self.calc_and_save_accuracy(new_predictions, user)

        # Otherwise, start updating all the predictions stored on Storage
        else: 
            # Create a tmp local file to store the predictions downloaded from Storage
            tmp_filename = self.create_tmp_filename()

            # Save the stored predictions to a tmp file
            pred_obj.download_to_filename(tmp_filename)
            
            # Load in memory
            predictions = pd.read_csv(tmp_filename)
            
            # Rename the new_predictions 'prediction' column to avoid merge conflicts
            new_predictions.rename(columns={'prediction': 'new_prediction'}, inplace=True)

            # Apply the new predictions on the old 
            # Merge the two datasets
            merged_df = pd.merge(predictions, new_predictions, how='outer', on='id')
            merged_df.set_index('id', inplace=True)
            
            # Update the old prediction when the new is there
            merged_df.loc[new_predictions.index, 'prediction'] = new_predictions['new_prediction']

            # Save to file and upload to Storage
            merged_df[['prediction', 'actual']].to_csv(tmp_filename)

            pred_obj.upload_from_filename(tmp_filename)

            # Delete tmp file
            os.remove(tmp_filename)

            # Calculate and save accuracy
            self.calc_and_save_accuracy(merged_df, user)

    def save_prediction_and_accuracy(self, prediction, user):
        '''
        Updates the predictions file with the provided prediction
        '''
        # Create a dataframe for that prediction
        new_predictions = pd.DataFrame(np.array([[prediction['id'], prediction['monthly']]]), columns=['id', 'monthly'])
        new_predictions.set_index("id", inplace=True)
        new_predictions.rename(columns={"monthly": "prediction"}, inplace=True)

        self.save(new_predictions=new_predictions, user=user)

    def save_predictions_and_accuracy(self, predictions_filename, user): 
        '''
        Saves the predictions to file, updating the file with the model's predictions 
        to support recalculation of the accuracy
        '''
        # Load predictions in pandas
        new_predictions = pd.read_csv(predictions_filename)[['id', 'occurs_monthly']]
        new_predictions.rename(columns={'occurs_monthly': 'prediction'}, inplace=True)
        new_predictions.set_index('id', inplace=True)

        self.save(new_predictions=new_predictions, user=user)

    def save_label_and_accuracy(self, label, user): 
        '''
        Saves a new label, recalculates accuracy and returns it
        Requrires "label" to be a dictionnary with: 'id', 'monthly'
        '''
        # Load the accuracy into pandas
        new_actuals = pd.DataFrame(np.array([[label['id'], label['monthly']]]), columns=['id', 'new_actual'])
        new_actuals.set_index('id', inplace=True)

        # Load the predictions file of this version fo the model
        pred_obj = self.get_bucket_object(user)

        # If there are no predictions, return
        if not pred_obj.exists():
            return None

        # Create a tmp local file to store the predictions downloaded from Storage
        tmp_filename = self.create_tmp_filename()

        # Save the stored predictions to a tmp file
        pred_obj.download_to_filename(tmp_filename)
        
        # Load in memory
        predictions = pd.read_csv(tmp_filename)
        
        # Apply the new actuals on the old 
        # Merge the two datasets
        merged_df = pd.merge(predictions, new_actuals, how='outer', on='id')
        merged_df.set_index('id', inplace=True)
        
        # Update the old prediction when the new is there
        merged_df.loc[new_actuals.index, 'actual'] = new_actuals['new_actual']

        # Save to file and upload to Storage
        merged_df[['prediction', 'actual']].to_csv(tmp_filename)

        pred_obj.upload_from_filename(tmp_filename)

        # Delete tmp file
        os.remove(tmp_filename)

        # Calculate and save accuracy
        return self.calc_and_save_accuracy(merged_df, user)        



