import os
import requests
import pandas as pd

from pandas import json_normalize
from toto_logger.logger import TotoLogger

toto_auth = os.environ['TOTO_API_AUTH']
toto_host = os.environ['TOTO_HOST']

logger = TotoLogger()

class HistoryDownloader: 

    def __init__(self, output_file_name, correlation_id): 
        self.empty = False
        self.output_file_name = output_file_name
        self.correlation_id = correlation_id

    def download(self, user, dateGte='20100101'): 
        '''
        This method downloads all historical movements and saves it as a temporary file, to be used to then feature
        engineer the data to infer
        '''
        response = requests.get(
            'https://{host}/apis/expenses/expenses?user={user}&dateGte={dateGte}'.format(user=user, dateGte=dateGte, host=toto_host),
            headers={
                'Accept': 'application/json',
                'Authorization': toto_auth,
                'x-correlation-id': self.correlation_id
            }
        )

        json_response = response.json()

        try: 
            expenses = json_response['expenses']
        except: 
            logger.compute(self.correlation_id, 'Error reading the following microservice response: {}'.format(json_response), 'error')
            self.empty = True
            return

        df = json_normalize(expenses)

        if df.empty: 
            self.empty = True
            return
        
        try: 
            df = df[['id', 'amount', 'category', 'date', 'description', 'monthly', 'yearMonth']]
        except KeyError: 
            df = df[['id', 'amount', 'category', 'date', 'description', 'yearMonth']]

        # Generate a new yearMonth column
        df['yearMonth'] = pd.to_numeric(df['date'].str[:-2])

        # Sort the dataframe
        df.sort_values(by=['date'], ascending=True, inplace=True)

        # Save the dataframe
        df.to_csv(self.output_file_name) 


