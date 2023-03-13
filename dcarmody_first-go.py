#!/usr/bin/env python
# coding: utf-8








# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')




train.info()




y = train['trip_duration']
#X = train[['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime',
#       'passenger_count', 'pickup_longitude', 'pickup_latitude',
#       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']]
X = train[['vendor_id',
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude']]




X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)




rf = RandomForestRegressor()




rf.fit(X_train,y_train)




pred = rf.predict(X_test)




pred




(mean_squared_log_error(y_test,pred))**0.5




test_pred = rf.predict(test[['vendor_id',
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude']])




submission = pd.DataFrame(list(zip(test['id'],test_pred)),columns=['id','trip_duration'])




submission.to_csv('submission.csv',index=None)




pd.read_csv('../working/submission.csv')






