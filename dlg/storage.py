import os
import pandas as pd
import uuid
from google.cloud import storage
from google.api_core.exceptions import NotFound
from toto_logger.logger import TotoLogger

client = storage.Client.from_service_account_json('/Users/nicolas/Developper/keys/toto-service-account-dev/toto-microservice-dev.json')

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

    def save_prediction(self, prediction, user):
        '''
        Updates the predictions file with the provided prediction
        '''

    def save_predictions(self, predictions_filename, user): 
        '''
        Saves the predictions to file, updating the file with the model's predictions 
        to support recalculation of the accuracy
        '''
        # Load predictions in pandas
        new_predictions = pd.read_csv(predictions_filename)[['id', 'occurs_monthly']]
        new_predictions.rename(columns={'occurs_monthly': 'prediction'}, inplace=True)
        new_predictions.set_index('id', inplace=True)

        # Load previous predictions from bucket, if any
        pred_obj_name = '{model}/{version}/predictions/{user}.predictions.csv'.format(model=self.model_name, version=self.model_version, user=user)
        pred_obj = self.bucket.blob(pred_obj_name)

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




