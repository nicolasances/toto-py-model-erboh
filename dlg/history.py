import os
import requests
import pandas as pd
import numpy as np
import datetime 
import dateutil.relativedelta

from pandas import json_normalize
from toto_logger.logger import TotoLogger

toto_auth = os.environ['TOTO_API_AUTH']
toto_host = os.environ['TOTO_HOST']

logger = TotoLogger()

class HistoryDownloader: 

    def __init__(self, folder, correlation_id): 
        self.correlation_id = correlation_id
        self.folder = folder

    def download_from(self, user, num_months, from_date): 
        """
        This method will download the historical data starting #num_months before #from_date

        Parameters
        ----------
        num_months (int) 
            The number of months to download, going back from the from_date argument

        from_date (string) formatted YYYYMMDD
            The date from which to start the download, going back #num_months
        """
        date_obj = datetime.datetime.strptime(from_date, '%Y%m%d')
        ym = str(date_obj.year) + str(date_obj.month).rjust(2, '0')

        # Find the right month
        date_obj = date_obj.replace(day=1)
        date_obj -= dateutil.relativedelta.relativedelta(months=num_months)

        dateGte = date_obj.strftime('%Y%m%d')

        return self.download(user, dateGte=dateGte)


    def download(self, user, dateGte='20100101'): 
        '''
        This method downloads all historical movements and saves it as a temporary file, to be used to then feature
        engineer the data to infer
        '''

        logger.compute(self.correlation_id, '[ HISTORICAL ] - Starting historical data download from date {date}'.format(date=dateGte), 'info')

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
            return None

        # Create a pandas data frame
        df = json_normalize(expenses)

        if df.empty: 
            logger.compute(self.correlation_id, '[ HISTORICAL ] - No historical data', 'warn')
            return None
        
        try: 
            df = df[['id', 'amount', 'category', 'date', 'description', 'monthly', 'yearMonth', 'user']]
        except KeyError: 
            logger.compute(self.correlation_id, '[ HISTORICAL ] - No "monthly" field found in the response! Skipping it!', 'warn')
            df = df[['id', 'amount', 'category', 'date', 'description', 'yearMonth', 'user']]

        # Generate a new yearMonth column
        df['yearMonth'] = pd.to_numeric(df['date'].str[:-2])

        # Sort the dataframe
        df.sort_values(by=['date'], ascending=True, inplace=True)

        # Save the dataframe
        df.to_csv(history_filename) 

        logger.compute(self.correlation_id, '[ HISTORICAL ] - Historical data downloaded: {} rows'.format(len(df)), 'info')

        # Return the filename
        return history_filename

    def append_expense(self, expense_id, user, amount, category, description, date): 
        """
        Appends the provided expense to the downloaded history file
        """
        logger.compute(self.correlation_id, '[ HISTORICAL ] - Appending expense to historical data.', 'info')

        history_filename = '{folder}/history.{user}.csv'.format(user=user, folder=self.folder);

        date_obj = datetime.datetime.strptime(date, '%Y%m%d')
        ym = str(date_obj.year) + str(date_obj.month).rjust(2, '0')

        # Read the historical data
        history_df = pd.read_csv(history_filename)

        # Create a new Data Frame for the expense to add
        new_df = pd.DataFrame(np.array([[expense_id, amount, category, date, description, None, ym, user]]), columns=['id','amount','category','date','description','monthly','yearMonth','user'])

        # Merge the df 
        full_df = pd.concat([history_df, new_df], axis=0, ignore_index=True, sort=False)

        # Save to a new file
        history_ext_filename = history_filename[:-4] + '.ext.csv'

        full_df.to_csv(history_ext_filename)

        return history_ext_filename

