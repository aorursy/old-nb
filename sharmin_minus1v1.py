#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

print ('Loading input files..')
print ()
people = pd.read_csv('../input/people.csv',
                       dtype={'people_id': np.str,
                              'activity_id': np.str,
                              'char_38': np.int32},
                       parse_dates=['date'])
train = pd.read_csv(r'../input/act_train.csv',
                        dtype={'people_id': np.str,
                               'activity_id': np.str,
                               'outcome': np.int8},
                        parse_dates=['date'])
test = pd.read_csv('../input/act_test.csv',
                       dtype={'people_id': np.str,
                              'activity_id': np.str},
                       parse_dates=['date'])

missing_values = []

print ('Train set features')
print ('------------------')
for col in train:
    unique = train[col].unique()
    print (str(col) + ' has ' + str(unique.size) + ' unique values')
    
    if (True in pd.isnull(unique)):
        print (str(col) + ' has ' + str(pd.isnull(train[col]).sum()) + ' missing values')
    print ()
    
print ()

print ('Processing the datasets..')
print ()
for data in [train,test]:
    for i in range(1,11):
        data['char_'+str(i)].fillna('type -1', inplace = 'true')
        data['char_'+str(i)] = data['char_'+str(i)].str.lstrip('type ').astype(np.int32)
        
    data['activity_category'] = data['activity_category'].str.lstrip('type ').astype(np.int32)
    
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data.drop('date', axis=1, inplace=True)
    
for i in range(1,10):
    people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
for i in range(10, 38):
    people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)
    
people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
people['year'] = people['date'].dt.year
people['month'] = people['date'].dt.month
people['day'] = people['date'].dt.day
people.drop('date', axis=1, inplace=True)

print ('Merging the datasets..')
print ()

train = pd.merge(train, people, how='left', on='people_id', left_index=True)
train.fillna(-1, inplace=True)
test = pd.merge(test, people, how='left', on='people_id', left_index=True)
test.fillna(-1, inplace=True)

train = train.drop(['people_id'], axis=1)

#Separate label and data
Y = train['outcome']
X = train.drop(['outcome'], axis=1)
X = X.iloc[:,1:]
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=96)

#print("cv")
#scores = cross_val_score(rfc, X, Y, cv=4)
#print ("Mean accuracy of Random Forest: " + scores.mean())
rfc = rfc.fit(X, Y)
#drop the people_id
test = test.drop(['people_id'], axis=1)
# Get the test data features, skipping the first column 'PassengerId'
test_x = test.iloc[:, 1:]


# Predict the outcome values for the test data
test_y = list(map(int, rfc.predict(test_x)))
#file for submission
test['outcome'] = test_y
test[['activity_id', 'outcome']]     .to_csv('results.csv', index=False)

