# Test of ML Flow

import re
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime as dt, timedelta
from sklearn.metrics import f1_score, confusion_matrix, classification_report

pd.options.mode.chained_assignment = None

def get_expenses_of_month(full_dataset, current_date, yearMonthDelta): 
    '''
    This function returns all the expenses of the specified month delta. 
    - yearMonthDelta:  an integer that specifies how many months before or after this one to look at (+1, +2, -1, ...)
    '''
    now = current_date

    currentMonth = now.month
    currentYear = now.year
    
    targetMonth = currentMonth
    targetYear = currentYear
    
    if (yearMonthDelta < 0): 
        targetMonth = currentMonth + yearMonthDelta + 12*int( (12 - (currentMonth + yearMonthDelta)) / 12 )
        targetYear -= int( (12 - (currentMonth + yearMonthDelta)) / 12 )
    else: 
        targetMonth = currentMonth + yearMonthDelta - 12*int( (currentMonth + yearMonthDelta - 1) / 12 )
        targetYear += int( (currentMonth + yearMonthDelta - 1) / 12 )
            
    yearMonth = int(str(targetYear) + str(targetMonth).rjust(2, '0'))
    
    return full_dataset[full_dataset['yearMonth'] == yearMonth]

def bow(df):
    # Remove all special characters
    df['description'] = df['description'].apply(lambda x : re.sub('[\.]', ' ', x))
    
    # Generate a bag of words for description
    vectorizer = CountVectorizer()
    vectorizer.fit(df['description'])

    df['description_bow'] = df['description'].apply(lambda x : vectorizer.transform([x]).toarray())
    
    return df

def overlap_word_count(row1, row2): 
    compared = np.logical_and(row1.tolist()[0], row2.tolist()[0])
    return np.sum(compared)

def same_amt_cat(row, dataset): 

    ds = dataset[(dataset['category'] == row['category']) & (dataset['amount'] == row['amount']) & (dataset['user'] == row['user'])]
    
    # Features
    same_amt_cat_sw = 0 # sw stands for share words
    same_amt_cat_sw_2 = 0
    same_amt_cat_sw_3more = 0
    same_amt_cat = 0
    same_amt_cat_2more = 0
    sacd = 0
    sacd3 = 0
    
    # If there are items with the same category and amount
    if len(ds) > 0:
        # Count the number of overlapping words in the description for all items that have the same cat and amt
        overlaps = ds['description_bow'].apply(overlap_word_count, row2=row['description_bow'])
        # Find the best match
        best_match = ds.loc[overlaps.idxmax(), :]
        # How many words in common?
        words_in_common = overlaps[overlaps.idxmax()]
        
        # Check if one of the items has the same exact date
        days = pd.to_datetime(ds['date'], format='%Y%m%d').dt.day
        row_day = dt.strptime(str(row['date']), '%Y%m%d').day
        if sum(days == row_day) > 0: 
            sacd = 1
        elif sum(days <= row_day + 3) > 0 and sum(days >= row_day - 3) > 0:
            sacd3 = 1
        
        # Is there exactly 1 payment with the same amt, cat and that shares words?
        if len(ds) == 1 and sum(overlaps > 0) == 1:
            same_amt_cat_sw = 1
        
        # Is there exactly 1 payment with the same amt, cat and that DOES NOT share words?
        if len(ds) == 1 and sum(overlaps > 0) == 0:
            same_amt_cat = 1
            
        if len(ds) > 1: 
            # Are there 2 or more payments with the same amt and cat that DO NOT share words?
            if sum(overlaps > 0) == 0:
                same_amt_cat_2more = 1
            # Are there 2 payments with the same amt and cat that share some words?
            elif sum(overlaps > 0) == 2:
                same_amt_cat_sw_2 = 1
            # Are there 3+ payments with the same amt and cat that share some words?
            elif sum(overlaps > 0) > 2:
                same_amt_cat_sw_3more = 1
        
    return pd.Series({'same_amt_cat_sw': same_amt_cat_sw, 'same_amt_cat_sw_2': same_amt_cat_sw_2, 'same_amt_cat_sw_3more': same_amt_cat_sw_3more, 'same_amt_cat': same_amt_cat, 'same_amt_cat_2more': same_amt_cat_2more, 'sacd': sacd, 'sacd3': sacd3})

def sesm(row, dataset): 
    
    ds = dataset[(dataset['category'] == row['category']) & (dataset['amount'] == row['amount']) & (dataset['user'] == row['user'])]
    
    if len(ds) > 1:
        overlaps = ds['description_bow'].apply(overlap_word_count, row2=row['description_bow'])
    
        if sum(overlaps > 0) > 1:
            return 1
    
    return 0

def engineer_features_for_month(month, df, training=False): 
    '''
    Engineers the features for the specified month:
    - month: the month in a YYYYMM format
    - df:   the whole data set

    IMPORTANT: only generates features for the expenses that haven't the 'monthly' field already set WHEN NOT TRAINING!!
    '''
    try: 
        if not training:
            features_m = df[(df['yearMonth'] == month) & (df['monthly'].isnull())]
        else: 
            features_m = df[(df['yearMonth'] == month)]
    except KeyError: 
        features_m = df[(df['yearMonth'] == month)]

    if features_m.empty: 
        return features_m
    
    year = int(str(month)[0:4])
    m = int(str(month)[4:])
    current_date = dt(year, m, 1)

    # Features of month - 1
    df_month_m1 = get_expenses_of_month(df, current_date, -1)
    df_month_m2 = get_expenses_of_month(df, current_date, -2)
    df_month_m3 = get_expenses_of_month(df, current_date, -3)
    df_month_m4 = get_expenses_of_month(df, current_date, -4)

    df_month_p1 = get_expenses_of_month(df, current_date, 1)

    fm1 = features_m.apply(same_amt_cat, axis=1, dataset=df_month_m1)
    fm2 = features_m.apply(same_amt_cat, axis=1, dataset=df_month_m2)
    fm3 = features_m.apply(same_amt_cat, axis=1, dataset=df_month_m3)
    fm4 = features_m.apply(same_amt_cat, axis=1, dataset=df_month_m4)
    
    features_m['sesm'] = features_m.apply(sesm, axis=1, dataset=features_m)
    
    features_m[['sacsw1_m1', 'sacsw2_m1', 'sacsw3m_m1', 'sac1_m1', 'sac2m_m1', 'sacd_m1', 'sacd3_m1']] = fm1[['same_amt_cat_sw', 'same_amt_cat_sw_2', 'same_amt_cat_sw_3more', 'same_amt_cat', 'same_amt_cat_2more', 'sacd', 'sacd3']]
    features_m[['sacsw1_m2', 'sacsw2_m2', 'sacsw3m_m2', 'sac1_m2', 'sac2m_m2', 'sacd_m2', 'sacd3_m2']] = fm2[['same_amt_cat_sw', 'same_amt_cat_sw_2', 'same_amt_cat_sw_3more', 'same_amt_cat', 'same_amt_cat_2more', 'sacd', 'sacd3']]
    features_m[['sacsw1_m3', 'sacsw2_m3', 'sacsw3m_m3', 'sac1_m3', 'sac2m_m3', 'sacd_m3', 'sacd3_m3']] = fm3[['same_amt_cat_sw', 'same_amt_cat_sw_2', 'same_amt_cat_sw_3more', 'same_amt_cat', 'same_amt_cat_2more', 'sacd', 'sacd3']]
    features_m[['sacsw1_m4', 'sacsw2_m4', 'sacsw3m_m4', 'sac1_m4', 'sac2m_m4', 'sacd_m4', 'sacd3_m4']] = fm4[['same_amt_cat_sw', 'same_amt_cat_sw_2', 'same_amt_cat_sw_3more', 'same_amt_cat', 'same_amt_cat_2more', 'sacd', 'sacd3']]
    
    return features_m

def category_dummies(cat): 
    if cat == 'SUPERMERCATO':
        return pd.Series([1, 0, 0, 0, 0, 0])
    elif cat == 'FOOD':
        return pd.Series([0, 1, 0, 0, 0, 0])
    elif cat == 'VIAGGI':
        return pd.Series([0, 0, 1, 0, 0, 0])
    elif cat == 'PALESTRA':
        return pd.Series([0, 0, 0, 1, 0, 0])
    elif cat == 'SALUTE':
        return pd.Series([0, 0, 0, 0, 1, 0])
    elif cat == 'XMAS':
        return pd.Series([0, 0, 0, 0, 0, 1])
    else:
        return pd.Series([0, 0, 0, 0, 0, 0])

class FeatureEngineering: 

    def __init__(self, data_file, output_file_name, training=False):
        self.output_file_name = output_file_name
        self.data_file = data_file
        self.model_feature_names = None
        self.empty = False
        self.count = 0
        self.training = training

    def do(self): 

        # Read all the data
        data = pd.read_csv(self.data_file)

        # Generate bow for descriptions
        data = bow(data)

        # Get the distinct months in the data
        months = data['yearMonth'].value_counts().index.sort_values()

        # Create the features data frame
        # This dataframe won't just contain features, but also needed references (e.g. id)
        features = pd.DataFrame()

        for month in months: 
            features = pd.concat([features, engineer_features_for_month(month, data, self.training)], ignore_index=True, sort=False)

        if features.empty:
            self.empty = True
            return

        # Finally: create dummies for the category
        features[['category_SUPERMERCATO', 'category_FOOD', 'category_VIAGGI', 'category_PALESTRA', 'category_SALUTE', 'category_XMAS']] = features['category'].apply(category_dummies)

        # Define the name of the features
        self.model_feature_names = ['sacsw1_m1', 'sacsw2_m1', 'sacsw3m_m1', 'sac1_m1', 'sac2m_m1', 'sacd_m1', 'sacd3_m1',
              'sacsw1_m2', 'sacsw2_m2', 'sacsw3m_m2', 'sac1_m2', 'sac2m_m2', 'sacd_m2', 'sacd3_m2',
              'sacsw1_m3', 'sacsw2_m3', 'sacsw3m_m3', 'sac1_m3', 'sac2m_m3', 'sacd_m3', 'sacd3_m3',
              'sacsw1_m4', 'sacsw2_m4', 'sacsw3m_m4', 'sac1_m4', 'sac2m_m4', 'sacd_m4', 'sacd3_m4', 
              'sesm', 
              'category_SUPERMERCATO', 'category_FOOD', 'category_VIAGGI', 'category_PALESTRA', 'category_SALUTE', 'category_XMAS'
              ]

        all_features_names = self.model_feature_names.copy()
        all_features_names.append('id')
        if self.training: 
            all_features_names.append('monthly')

        # Save all the features to file
        features[all_features_names].to_csv(self.output_file_name)

        # Save additional data
        self.count = len(features)