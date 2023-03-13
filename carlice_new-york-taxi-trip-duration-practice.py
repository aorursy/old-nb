#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import seaborn as sns
from ggplot import *
from sklearn.cross_validation import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')




df1 = pd.read_csv('../input/train.csv')
df2 = pd.read_csv('../input/test.csv')
df3 = pd.read_csv('../input/sample_submission.csv')




df2['trip_duration'] = df3['trip_duration']




print (df1.count())
print ('-----------------------------')
print (df2.count())




#combine the train dataset and the test dataset and then split the dataset
df = df1.append(df2, ignore_index= True)
df = df[['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
        'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'trip_duration']]
df.head(5)




df.count()




df.info()




#facet plots with ggplot
ggplot(df, aes(x='pickup_longitude', y='trip_duration', color='store_and_fwd_flag')) +     geom_point(size=30, shape=4) + facet_wrap('vendor_id') + xlim(-80, -60) + ylim(0, 20000) 




#heatmap
corr = df.corr()
heatmap = sns.heatmap(corr, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, cmap = 'YlGnBu')




# create test and training sets 
x = df.drop('trip_duration', axis = 1)
y = df.drop(df[:10], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=None)
print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)






