import pandas as pd

from toto_pubsub.publisher import TotoEventPublisher
from google.cloud import pubsub_v1

publisher = TotoEventPublisher(microservice='model-erboh', topics=['expenseUpdateRequested'])

class ExpenseUpdater: 

    def __init__(self, correlation_id):
        self.correlation_id = correlation_id

    def do(self, predictions_filename=None, expense=None): 
        '''
        Updates expenses. The input can be either a file or a single expense.
        In case of an expense, it requires a dictionnary: {id: string, monthly: float (1 or 0)}
        '''
        if predictions_filename: 
            self.doFile(predictions_filename)
        else: 
            self.doSingle(expense)

    def doSingle(self, expense):
        '''
        Updates a single expense
        '''

        id = expense['id']

        if expense['monthly'] == 1:
            monthly = True
        else: 
            monthly = False
        
        # Format the message
        msg = {
            'correlationId': self.correlation_id,
            "id": id, 
            "monthly": monthly
        }

        # Post the expense to the update queue
        publisher.publish(topic='expenseUpdateRequested', event=msg)

    def doFile(self, predictions_filename):
        '''
        Updates expenses
        The input is a file
        '''

        # Load the predictions
        predictions = pd.read_csv(predictions_filename)

        for index, row in predictions.iterrows():
            id = row['id']
            if row['occurs_monthly'] == 1: 
                monthly = True
            else: 
                monthly = False

            # Format the message
            msg = {
                'correlationId': self.correlation_id,
                "id": id, 
                "monthly": monthly
            }

            # Post the expense to the update queue
            publisher.publish(topic='expenseUpdateRequested', event=msg)
