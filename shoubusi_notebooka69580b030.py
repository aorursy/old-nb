#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


from time import time


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png' #set 'png' here when working on notebook")
get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




train.tail()




targets = train.columns.tolist()[-3:]
feats = train.columns.tolist()[:-3]

feature_data = pd.concat((train[feats],
                          test[feats]))

feature_data.datetime = pd.to_datetime(feature_data.datetime)

feature_data['hour'] = feature_data.datetime.dt.hour
feature_data['day'] = feature_data.datetime.dt.day
feature_data['dayofweek'] = feature_data.datetime.dt.dayofweek
feature_data.drop('datetime', axis=1, inplace=True)

feats = ['hour', 'day', 'dayofweek', 'season', 'holiday', 
         'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed']

feature_data = pd.DataFrame(feature_data, columns=feats)




matplotlib.rcParams['figure.figsize'] = (8, 8)
feature_data.hist(bins=40)




feature_data['windspeed'] = np.log1p(feature_data['windspeed'])
X_train = feature_data[:train.shape[0]]
X_test = feature_data[train.shape[0]:]
Y_train = np.log1p(train.loc[:, 'casual':'count'])




Y_train.hist(bins=30)




from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from time import time




# cat ../input/sampleSubmission.csv

