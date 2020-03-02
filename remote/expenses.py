import pandas as pd

from toto_pubsub.publisher import TotoEventPublisher
from google.cloud import pubsub_v1
from toto_logger.logger import TotoLogger

publisher = TotoEventPublisher(microservice='model-erboh', topics=['expenseUpdateRequested'])

logger = TotoLogger()

def update_expense(expense, correlation_id): 
    """
    This method updates a single expense
    It updates the "monthly" property of the expense
    """
    logger.compute(correlation_id, '[ UPDATE ] - Updating payment with prediction', 'info')

    id = expense['id']

    if expense['monthly'] == 1:
        monthly = True
    else: 
        monthly = False
    
    # Format the message
    msg = {
        'correlationId': correlation_id,
        "id": id, 
        "monthly": monthly
    }

    # Post the expense to the update queue
    publisher.publish(topic='expenseUpdateRequested', event=msg)

    logger.compute(correlation_id, '[ UPDATE ] - Payment updated', 'info')

def update_expenses(predictions_filename, correlation_id): 
    """
    This method updates multiple expenses
    The input is a predictions filename
    """
    # Load the predictions
    predictions = pd.read_csv(predictions_filename)

    logger.compute(correlation_id, '[ UPDATE ] - Updating {} payments with predictions'.format(len(predictions)), 'info')

    for index, row in predictions.iterrows():
        id = row['id']
        if row['occurs_monthly'] == 1: 
            monthly = True
        else: 
            monthly = False

        # Format the message
        msg = {
            'correlationId': correlation_id,
            "id": id, 
            "monthly": monthly
        }

        # Post the expense to the update queue
        publisher.publish(topic='expenseUpdateRequested', event=msg)

    logger.compute(correlation_id, '[ UPDATE ] - Payments updated', 'info')
    
