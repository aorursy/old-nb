import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import log_loss
datadir = '../input'

gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),

                      index_col='device_id')

gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),

                     index_col = 'device_id')

phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

# Get rid of duplicate device ids in phone

phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')

events = pd.read_csv(os.path.join(datadir,'events.csv'),

                     parse_dates=['timestamp'], index_col='event_id')

appevents = pd.read_csv(os.path.join(datadir,'app_events.csv'), 

                        usecols=['event_id','app_id','is_active'],

                        dtype={'is_active':bool})

applabels = pd.read_csv(os.path.join(datadir,'app_labels.csv'))
gatrain['trainrow']=np.arange(gatrain.shape[0])

gatest['testrow']=np.arange(gatest.shape[0])
len(gatrain)

len(gatest)
brandencoder=LabelEncoder().fit(phone.phone_brand)

phone['brand'] = brandencoder.transform(phone['phone_brand'])

gatrain['brand'] = phone['brand']

gatrain.head(20)