#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 09:51:49 2018

@author: canf
"""

import numpy as np 
import pandas as pd

# Load file to dataframe using pandas
# Using column 'row_id' to match train and test files
def load_data(filename):
    df_val = pd.read_csv(filename, index_col=['id'])
    return df_val

# Using function to load data files
filename = './input/train_values.csv'
train = load_data(filename)
filename = './input/test_values.csv'
test = load_data(filename)

num_renewal = train['renewal'].value_counts()
numPositives = num_renewal.tolist()[1]
numNegatives = num_renewal.tolist()[0]
totalNumSamples = len(train)
print('Total number of data samples: {}'.format(totalNumSamples))
print('Non-Renewal percentage (%): {0:2.5f}%'.format(numPositives/totalNumSamples*100))
print('Renewal percentage (%): {0:2.5f}%'.format(numNegatives/totalNumSamples*100))
print(' ')

train_na = (train.isnull().sum() / len(train)) * 100
train_na = \
    train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :train_na})
print('Train Data Missing Ratio')
print(missing_data.head(15))
print(' ')

test_na = (test.isnull().sum() / len(test)) * 100
test_na = \
    test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :test_na})
print('Test Data Missing Ratio')
print(missing_data.head(15))
print(' ')

for dataset in [train]:
    dataset['Count_more_than_12_months_late'].fillna(0, inplace=True)
    dataset['Count_6-12_months_late'].fillna(0, inplace=True)
    dataset['Count_3-6_months_late'].fillna(0, inplace=True)
    dataset['application_underwriting_score'].fillna( \
        dataset['application_underwriting_score'].median(), inplace=True)
    
for dataset in [test]:
    dataset['Count_more_than_12_months_late'].fillna(0, inplace=True)
    dataset['Count_6-12_months_late'].fillna(0, inplace=True)
    dataset['Count_3-6_months_late'].fillna(0, inplace=True)
    dataset['application_underwriting_score'].fillna( \
        dataset['application_underwriting_score'].median(), inplace=True)
    
train['sourcing_channel'] = train['sourcing_channel'].astype('category')
train['residence_area_type'] = train['residence_area_type'].astype('category')

test['sourcing_channel'] = test['sourcing_channel'].astype('category')
test['residence_area_type'] = test['residence_area_type'].astype('category')

# Create a set of dummy variables from the categorical variables
train_sourcing = pd.get_dummies(train['sourcing_channel'])
train_residence = pd.get_dummies(train['residence_area_type'])
# Join the dummy variables to the main dataframe
train_new = pd.concat([train, train_sourcing, train_residence], axis=1)
train_df = pd.DataFrame(train_new)

# Create a set of dummy variables from the categorical variables
test_sourcing = pd.get_dummies(test['sourcing_channel'])
test_residence = pd.get_dummies(test['residence_area_type'])
# Join the dummy variables to the main dataframe
test_new = pd.concat([test, test_sourcing, test_residence], axis=1)
test_df = pd.DataFrame(test_new)

train_df = train_df.drop(['sourcing_channel', 'residence_area_type'], axis=1)
test_df = test_df.drop(['sourcing_channel', 'residence_area_type'], axis=1)

print(train_df.dtypes)
print(test_df.dtypes)

X = list(train_df.columns.values)
print(X)
y = 'renewal'
print(y)

import h2o
h2o.init(max_mem_size_GB=14)

#####################
#   DATA LOADING    #
#####################

# Load data using h2o
# train_data = h2o.import_file(train_new)
train_data = h2o.H2OFrame(train_df)
print('Data loaded successfully')
print(' ')

#######################
#   MODEL TRAINING    #
#######################

# Create classifier, balance classes
clf = h2o.estimators.random_forest.H2ORandomForestEstimator(balance_classes=True)
print('Training model...')
print(' ')

# Train classifier, delete train data afterwards
clf.train(X, y, training_frame=train_data)
print('Model trained!')
print(' ')

# Check model performance
clf.model_performance()
clf.auc(valid=True)

###################
#   Submission    #
###################

# Import test dataset
test_data = h2o.H2OFrame(test_df)

# Process dataframe in place
# process_data(test_data)

# Perform predictions
pred = clf.predict(test_data)

# Create the submission file
submission = pd.read_csv('./input/sample_submission.csv')
submission[y] = h2o.as_list(pred)
submission.to_csv('h2o_randomforest.csv', index=False)