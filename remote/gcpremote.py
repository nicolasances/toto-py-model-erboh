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

def save_retrained_model_pickle(model_name, model_object):
    """
    Saves the pickle file of the retrained model
    """

    try:
        bucket = client.get_bucket(bucket_name)
    except NotFound: 
        logger.compute('no-id', 'Bucket {} not found. Please create it first'.format(bucket_name), 'error')
        return

    # Store the model on a tmp file
    filename = uuid.uuid1()
    target_file = "{folder}/{file}".format(folder=base_folder, file=filename)

    joblib.dump(model_object, target_file)

    # Upload to bucket
    bucket_objname = '{model}/retrained/{model}'.format(model=model_name)
    bucket_obj = bucket.blob(bucket_objname)

    bucket_obj.upload_from_filename(target_file)

def init_champion_model(model, cid):
    """
    Initializes the pickle file of the champion model by doing the following:
     - checks if there's a champion model available in the bucket's /champion folder
     - if not, uploads the default erboh.v1 model found locally
    
    Parameters
    ----------
    model (dict)
        The loaded model (see function check_registry, that returns the champion model)
        The following fields are needed: 
        - name (string): the name of the model
        - version (int): the version of the model
    """

    try:
        bucket = client.get_bucket(bucket_name)
    except NotFound: 
        logger.compute(cid, 'Bucket {} not found. Please create it first'.format(bucket_name), 'error')
        return

    bucket_objname = '{model}/champion/{model}.v{version}'.format(model=model['name'], version=model['version'])
    bucket_obj = bucket.blob(bucket_objname)

    if bucket_obj.exists():
        logger.compute(cid, 'Champion Model {model}.v{version} found in the bucket'.format(model=model['name'], version=model['version']), 'info')
        return 
    
    # If the object does not exist, upload erboh.v1
    logger.compute(cid, 'Champion Model {model}.v{version} NOT found in the bucket. Creating it.'.format(model=model['name'], version=model['version']), 'info')

    bucket_obj.upload_from_filename('erboh.v1')

    logger.compute(cid, 'Champion Model {model}.v{version} created.'.format(model=model['name'], version=model['version']), 'info')

def load_champion_model(model_info, cid): 
    """ 
    Loads the champion model from the bucket

    Parameters
    ----------
    model_info (dict)
        Contains the "name" and "version" of the model
        Used to look for the model pickle file
    """
    try:
        bucket = client.get_bucket(bucket_name)
    except NotFound: 
        logger.compute(cid, 'Bucket {} not found. Please create it first'.format(bucket_name), 'error')
        return

    bucket_objname = '{model}/champion/{model}.v{version}'.format(model=model_info['name'], version=model_info['version'])
    bucket_obj = bucket.blob(bucket_objname)

    if bucket_obj.exists():
        # Store locally
        target_file = "{folder}/{file}".format(folder=base_folder, file=uuid.uuid1())
        
        bucket_obj.download_to_filename(target_file)

        return joblib.load(target_file)

    return None
