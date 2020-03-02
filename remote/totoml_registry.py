import os 
import requests
import datetime
from toto_logger.logger import TotoLogger
from random import randint

toto_auth = os.environ['TOTO_API_AUTH']
toto_host = os.environ['TOTO_HOST']

logger = TotoLogger()

def cid(): 
    '''
    Generates a Toto-valid Correlation ID
    Example: 20200229160021694-09776
    '''
    datepart = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    randpart = str(randint(0, 100000)).zfill(5)

    return '{date}-{rand}'.format(date=datepart, rand=randpart)

def check_registry(model_name): 
    '''
    Checks if the model exists in the Toto ML Model Registry. 
    If it doesn't it will create it. 
    '''
    correlation_id = cid()

    logger.compute(correlation_id, 'Checking if model [{model}] exists in Toto ML Model Registry'.format(model=model_name), 'info')

    response = requests.get(
        'https://{host}/apis/totoml/registry/models/{name}'.format(host=toto_host, name=model_name),
        headers={
            'Accept': 'application/json',
            'Authorization': toto_auth,
            'x-correlation-id': cid()
        }
    )

    model = response.json()

    # Check if the model exists
    if not 'name' in model: 
        corr_id = cid()

        # If it doesn't, create the model
        model = {
            "name": model_name
        }

        logger.compute(correlation_id, 'Model [{model}] does not exist in Toto ML Model Registry. Creating it.'.format(model=model_name), 'info')

        response = requests.post(
            'https://{host}/apis/totoml/registry/models'.format(host=toto_host),
            headers={
                'Accept': 'application/json',
                'Authorization': toto_auth,
                'x-correlation-id': corr_id
            }, 
            json=model
        )

        if response.status_code != 201: 
            logger.compute(corr_id, 'Something went wrong when creating a new model on Toto ML Model Registry. Response: {content}'.format(content=response.content), 'error')
        else: 
            logger.compute(correlation_id, 'Model [{model}] created in Toto ML Model Registry'.format(model=model_name), 'info')
    else:
        logger.compute(correlation_id, 'Model [{model}] exists in Toto ML Model Registry'.format(model=model_name), 'info')


def post_retrained_model(model_name, metrics, corr_id): 
    """
    Posts a new retrained model to Toto ML Registry for a given model
    """

    response = requests.post(
        'https://{host}/apis/totoml/registry/models/{model}/retrained'.format(host=toto_host, model=model_name),
        headers={
            'Accept': 'application/json',
            'Authorization': toto_auth,
            'x-correlation-id': corr_id
        }, 
        json={
            "metrics": metrics
        }
    )

    if response.status_code != 201: 
        logger.compute(corr_id, 'Something went wrong when creating a new retrained model on Toto ML Model Registry for model {model}. Response: {content}'.format(model=model_name, content=response.content), 'error')

    logger.compute(corr_id, '[ POSTING ] - Posted retrained model', 'info')

    return

def put_champion_metrics(model_name, metrics, cid): 
    """
    Updates the metrics of the chamption model 

    Parameters
    ----------
    model_name (string)
        The name of the models

    metrics (list)
        A list [] of metrics object. 
        Each metric is a {name: <name>, value: <value>} object
    """
    response = requests.put(
        'https://{host}/apis/totoml/registry/models/{model}/metrics'.format(host=toto_host, model=model_name),
        headers={
            'Accept': 'application/json',
            'Authorization': toto_auth,
            'x-correlation-id': cid
        }, 
        json={
            "metrics": metrics
        }
    )

    if response.status_code != 200: 
        logger.compute(cid, 'Something went wrong when updating the metrics on Toto ML Model Registry for model {model}. Response: {content}'.format(model=model_name, content=response.content), 'error')

    logger.compute(cid, '[ POSTING ] - Updated metrics of champion model', 'info')

    return 

