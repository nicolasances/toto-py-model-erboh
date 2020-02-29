import os
import uuid
import joblib
from google.cloud import storage
from google.api_core.exceptions import NotFound
from toto_logger.logger import TotoLogger
from sklearn.metrics import precision_recall_fscore_support

client = storage.Client.from_service_account_json('/Users/nicolas/Developper/keys/toto-service-account-dev/toto-microservice-dev.json')

toto_env = os.environ['TOTO_ENV']
base_folder = os.environ['TOTO_TMP_FOLDER']

bucket_name = 'toto-{env}-model-storage'.format(env=toto_env)
os.makedirs(name=base_folder, exist_ok=True)

logger = TotoLogger()

def save_challenger(model_name, challenger_id, model_object):

    try:
        bucket = client.get_bucket(bucket_name)
    except NotFound: 
        logger.compute('no-id', 'Bucket {} not found. Please create it first'.format(bucket_name), 'error')
        return
    
    # Store the model on a tmp file
    target_file = "{folder}/{file}".format(folder=base_folder, file=challenger_id)

    joblib.dump(model_object, target_file)

    # Upload to bucket
    bucket_objname = '{model}/challengers/{chid}'.format(model=model_name, chid=challenger_id)
    bucket_obj = bucket.blob(bucket_objname)

    bucket_obj.upload_from_filename(target_file)

