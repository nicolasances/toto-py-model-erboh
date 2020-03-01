import os
import requests
import pandas as pd

from pandas import json_normalize
from toto_logger.logger import TotoLogger

toto_auth = os.environ['TOTO_API_AUTH']
toto_host = os.environ['TOTO_HOST']

logger = TotoLogger()

class HistoryDownloader: 

    def __init__(self, folder, correlation_id): 
        self.correlation_id = correlation_id
        self.folder = folder

    def download(self, user, dateGte='20100101'): 
        '''
        This method downloads all historical movements and saves it as a temporary file, to be used to then feature
        engineer the data to infer
        '''

        logger.compute(self.correlation_id, '[ HISTORICAL ] - Starting historical data download', 'info')

        history_filename = '{folder}/history.{user}.csv'.format(user=user, folder=self.folder);

        # Call the API to download the data
        response = requests.get(
            'https://{host}/apis/expenses/expenses?user={user}&dateGte={dateGte}'.format(user=user, dateGte=dateGte, host=toto_host),
            headers={
                'Accept': 'application/json',
                'Authorization': toto_auth,
                'x-correlation-id': self.correlation_id
            }
        )

        # Convert to JSON
        json_response = response.json()

        # Extract the expenses array
        try: 
            expenses = json_response['expenses']
        except: 
            logger.compute(self.correlation_id, '[ HISTORICAL ] - Error reading the following microservice response: {}'.format(json_response), 'error')
            logger.compute(self.correlation_id, '[ HISTORICAL ] - No historical data', 'warn')
            return

        # Create a pandas data frame
        df = json_normalize(expenses)

        if df.empty: 
            logger.compute(self.correlation_id, '[ HISTORICAL ] - No historical data', 'warn')
            return
        
        try: 
            df = df[['id', 'amount', 'category', 'date', 'description', 'monthly', 'yearMonth', 'user']]
        except KeyError: 
            df = df[['id', 'amount', 'category', 'date', 'description', 'yearMonth', 'user']]

        # Generate a new yearMonth column
        df['yearMonth'] = pd.to_numeric(df['date'].str[:-2])

        # Sort the dataframe
        df.sort_values(by=['date'], ascending=True, inplace=True)

        # Save the dataframe
        df.to_csv(history_filename) 

        logger.compute(self.correlation_id, '[ HISTORICAL ] - Historical data downloaded', 'info')

        # Return the filename
        return history_filename


