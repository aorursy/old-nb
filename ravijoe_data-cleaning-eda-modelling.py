#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.




train = pd.read_csv("../input/new-york-city-taxi-fare-prediction/train.csv", nrows = 1000000)
test = pd.read_csv("../input/new-york-city-taxi-fare-prediction/test.csv")




train.head()




test.shape




train.info()




train.describe()




#check for missing values in train data
train.isnull().sum().sort_values(ascending=False)




#check for missing values in test data
test.isnull().sum().sort_values(ascending=False)




#drop the missing values
train = train.drop(train[train.isnull().any(1)].index, axis = 0)




train.shape




#check the target column
train['fare_amount'].describe()




train = train.drop(train[train['fare_amount']<0].index, axis=0)
train.shape




#no more negative values in the fare field
train['fare_amount'].describe()




train['passenger_count'].describe()




# lets assume that in a suv maximum number of passengers can be 6 . still 208 passengers is unrealistic
train[train['passenger_count']>6]




train = train.drop(train[train['passenger_count']==208].index, axis = 0)




#much neater now! Max number of passengers are 6. Which makes sense i the cab is an SUV :)
train['passenger_count'].describe()




#Next, let us explore the pickup latitude and longitudes
train['pickup_latitude'].describe()




train[train['pickup_latitude']<-90]




train[train['pickup_latitude']>90]




#We need to drop these outliers
train = train.drop(((train[train['pickup_latitude']<-90])|(train[train['pickup_latitude']>90])).index, axis=0)




#12 rows dropped
train.shape




#similar operation for pickup longitude
train['pickup_longitude'].describe()




train[train['pickup_longitude']<-180]




train[train['pickup_longitude']>180]




train = train.drop(((train[train['pickup_longitude']<-180])|(train[train['pickup_longitude']>180])).index, axis=0)




#11 rows dropped
train.shape




#similar operation for dropoff latitude and longitude
train[train['dropoff_latitude']<-90]




train[train['dropoff_latitude']>90]




train = train.drop(((train[train['dropoff_latitude']<-90])|(train[train['dropoff_latitude']>90])).index, axis=0)




#8 rows dropped
train.shape




train[train['dropoff_latitude']<-180]|train[train['dropoff_latitude']>180]




train['key'] = pd.to_datetime(train['key'])
train['pickup_datetime']  = pd.to_datetime(train['pickup_datetime'])




#Convert for test data
test['key'] = pd.to_datetime(test['key'])
test['pickup_datetime']  = pd.to_datetime(test['pickup_datetime'])




#check the dtypes after conversion
train.dtypes




test.dtypes




#check the data
train.head()




test.head()








# we can also use the following code for haversine distance calculation
# from haversine import haversine, Unit

# lyon = (45.7597, 4.8422) # (lat, lon)
# paris = (48.8567, 2.3508)

# haversine(lyon, paris)
# >> 392.2172595594006  # in kilometers

# haversine(lyon, paris, unit=Unit.MILES)
# >> 243.71201856934454  # in miles

# # you can also use the string abbreviation for units:
# haversine(lyon, paris, unit='mi')
# >> 243.71201856934454  # in miles

# haversine(lyon, paris, unit=Unit.NAUTICAL_MILES)
# >> 211.78037755311516  # in nautical miles




def haversine_distance(lat1, long1, lat2, long2):
    data = [train, test]
    for i in data:
        R = 6371  #radius of earth in kilometers
        #R = 3959 #radius of earth in miles
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        i['H_Distance'] = d
    return d




haversine_distance('pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')




train['H_Distance'].head(10)




test['H_Distance'].head(10)




train.head(10)




test.head(10)




data = [train,test]
for i in data:
    i['Year'] = i['pickup_datetime'].dt.year
    i['Month'] = i['pickup_datetime'].dt.month
    i['Date'] = i['pickup_datetime'].dt.day
    i['Day of Week'] = i['pickup_datetime'].dt.dayofweek
    i['Hour'] = i['pickup_datetime'].dt.hour




train.head()




test.head()




plt.figure(figsize=(15,7)) 
sns.barplot(x="Year", y="fare_amount", data=train)




plt.figure(figsize=(15,7))
plt.hist(train['passenger_count'], bins=15)
plt.xlabel('No. of Passengers')
plt.ylabel('Frequency')




plt.figure(figsize=(15,7))
plt.scatter(x=train['passenger_count'], y=train['fare_amount'], s=1.5)
plt.xlabel('No. of Passengers')
plt.ylabel('Fare')




plt.figure(figsize=(15,7))
plt.scatter(x=train['Date'], y=train['fare_amount'], s=1.5)
plt.xlabel('Date')
plt.ylabel('Fare')




plt.figure(figsize=(15,7))

sns.boxplot(x="Date", y="fare_amount", data=train)




plt.figure(figsize=(15,7))
plt.hist(train['Hour'], bins=100)
plt.xlabel('Hour')
plt.ylabel('Frequency')




plt.figure(figsize=(15,7))
plt.scatter(x=train['Hour'], y=train['fare_amount'], s=1.5)
plt.xlabel('Hour')
plt.ylabel('Fare')




plt.figure(figsize=(15,7))
sns.barplot(x="Hour", y="fare_amount", data=train)




plt.figure(figsize=(15,7))
sns.distplot(train['Day of Week'],bins=100)




plt.figure(figsize=(15,7))
# plt.scatter(x=train['Day of Week'], y=train['fare_amount'], s=1.5)
sns.barplot(x='Day of Week', y="fare_amount", data=train)
plt.xlabel('Day of Week')
plt.ylabel('Fare')




train.sort_values(['H_Distance','fare_amount'], ascending=False)




plt.figure(figsize=(17,8))
sns.regplot(x="H_Distance", y="fare_amount", data=train);




from scipy.stats import pearsonr
pearsonr(train['H_Distance'], train['fare_amount'])




bins_0 = train.loc[(train['H_Distance'] == 0), ['H_Distance']]
bins_1 = train.loc[(train['H_Distance'] > 0) & (train['H_Distance'] <= 10),['H_Distance']]
bins_2 = train.loc[(train['H_Distance'] > 10) & (train['H_Distance'] <= 50),['H_Distance']]
bins_3 = train.loc[(train['H_Distance'] > 50) & (train['H_Distance'] <= 100),['H_Distance']]
bins_4 = train.loc[(train['H_Distance'] > 100) & (train['H_Distance'] <= 200),['H_Distance']]
bins_5 = train.loc[(train['H_Distance'] > 200) & (train['H_Distance'] <= 300),['H_Distance']]
bins_6 = train.loc[(train['H_Distance'] > 300),['H_Distance']]
bins_0['bins'] = '0'
bins_1['bins'] = '0-10'
bins_2['bins'] = '11-50'
bins_3['bins'] = '51-100'
bins_4['bins'] = '100-200'
bins_5['bins'] = '201-300'
bins_6['bins'] = '>300'
dist_bins =pd.concat([bins_0,bins_1,bins_2,bins_3,bins_4,bins_5,bins_6])
#len(dist_bins)
dist_bins.columns




Counter(dist_bins['bins'])




#pickup latitude and longitude = 0
train.loc[((train['pickup_latitude']==0) & (train['pickup_longitude']==0))&((train['dropoff_latitude']!=0) & (train['dropoff_longitude']!=0)) & (train['fare_amount']==0)]




train = train.drop(train.loc[((train['pickup_latitude']==0) & (train['pickup_longitude']==0))&((train['dropoff_latitude']!=0) & (train['dropoff_longitude']!=0)) & (train['fare_amount']==0)].index, axis=0)




#1 row dropped
train.shape




#Check in test data
test.loc[((test['pickup_latitude']==0) & (test['pickup_longitude']==0))&((test['dropoff_latitude']!=0) & (test['dropoff_longitude']!=0))]
#No records! PHEW!




#dropoff latitude and longitude = 0
train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))&((train['dropoff_latitude']==0) & (train['dropoff_longitude']==0)) & (train['fare_amount']==0)]




train = train.drop(train.loc[((train['pickup_latitude']!=0) & (train['pickup_longitude']!=0))&((train['dropoff_latitude']==0) & (train['dropoff_longitude']==0)) & (train['fare_amount']==0)].index, axis=0)




#3 rows dropped
train.shape




#Checking test data
#Again no records! AWESOME!
test.loc[((test['pickup_latitude']!=0) & (test['pickup_longitude']!=0))&((test['dropoff_latitude']==0) & (test['dropoff_longitude']==0))]




high_distance = train.loc[(train['H_Distance']>200)&(train['fare_amount']!=0)]




high_distance




high_distance.shape




high_distance['H_Distance'] = high_distance.apply(
    lambda row: (row['fare_amount'] - 2.50)/1.56,
    axis=1
)




#The distance values have been replaced by the newly calculated ones according to the fare
high_distance




#sync the train data with the newly computed distance values from high_distance dataframe
train.update(high_distance)




train.shape




train[train['H_Distance']==0]




train[(train['H_Distance']==0)&(train['fare_amount']==0)]




train = train.drop(train[(train['H_Distance']==0)&(train['fare_amount']==0)].index, axis = 0)




#4 rows dropped
train[(train['H_Distance']==0)].shape




#Between 6AM and 8PM on Mon-Fri
rush_hour = train.loc[(((train['Hour']>=6)&(train['Hour']<=20)) & ((train['Day of Week']>=1) & (train['Day of Week']<=5)) & (train['H_Distance']==0) & (train['fare_amount'] < 2.5))]
rush_hour




train=train.drop(rush_hour.index, axis=0)




train.shape




#Between 8PM and 6AM on Mon-Fri
non_rush_hour = train.loc[(((train['Hour']<6)|(train['Hour']>20)) & ((train['Day of Week']>=1)&(train['Day of Week']<=5)) & (train['H_Distance']==0) & (train['fare_amount'] < 3.0))]
#print(Counter(non_work_hours['Hour']))
#print(Counter(non_work_hours['Day of Week']))
non_rush_hour
#keep these. Since the fare_amount is not <2.5 (which is the base fare), these values seem legit to me.




#Saturday and Sunday all hours
weekends = train.loc[((train['Day of Week']==0) | (train['Day of Week']==6)) & (train['H_Distance']==0) & (train['fare_amount'] < 3.0)]
weekends
#Counter(weekends['Day of Week'])
#keep these too. Since the fare_amount is not <2.5, these values seem legit to me.




train.loc[(train['H_Distance']!=0) & (train['fare_amount']==0)]




scenario_3 = train.loc[(train['H_Distance']!=0) & (train['fare_amount']==0)]




len(scenario_3)




#We do not have any distance values that are outliers.
scenario_3.sort_values('H_Distance', ascending=False)




scenario_3['fare_amount'] = scenario_3.apply(
    lambda row: ((row['H_Distance'] * 1.56) + 2.50), axis=1
)




scenario_3['fare_amount']




train.update(scenario_3)




train.shape




train.loc[(train['H_Distance']==0) & (train['fare_amount']!=0)]




scenario_4 = train.loc[(train['H_Distance']==0) & (train['fare_amount']!=0)]




len(scenario_4)




#Using our prior knowledge about the base price during weekdays and weekends for the cabs.
#I do not want to impute these 1502 values as they are legible ones.
scenario_4.loc[(scenario_4['fare_amount']<=3.0)&(scenario_4['H_Distance']==0)]




scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]




scenario_4_sub = scenario_4.loc[(scenario_4['fare_amount']>3.0)&(scenario_4['H_Distance']==0)]




len(scenario_4_sub)




scenario_4_sub['H_Distance'] = scenario_4_sub.apply(
lambda row: ((row['fare_amount']-2.50)/1.56), axis=1
)




train.update(scenario_4_sub)




train.shape




train.columns




test.columns




#not including the pickup_datetime columns as datetime columns cannot be directly used while modelling. Features need to extracted from the 
#timestamp fields which will later be used as features for modelling.
train = train.drop(['key','pickup_datetime'], axis = 1)
test = test.drop(['key','pickup_datetime'], axis = 1)




train.columns




test.columns




x_train = train.iloc[:,train.columns!='fare_amount']
y_train = train['fare_amount'].values
x_test = test




x_train.shape




x_train.columns




y_train.shape




x_test.shape




x_test.columns




import lightgbm as lgbm




params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'verbose': 0,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_aplha': 1,
        'reg_lambda': 0.001,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1     
    }




pred_test_y = np.zeros(x_test.shape[0])
pred_test_y.shape




train_set = lgbm.Dataset(x_train, y_train, silent=True)
train_set




model = lgbm.train(params, train_set = train_set, num_boost_round=300)




pred_test_y = model.predict(x_test, num_iteration = model.best_iteration)




print(pred_test_y)




submission = pd.read_csv('../input/new-york-city-taxi-fare-prediction/sample_submission.csv')
submission['fare_amount'] = pred_test_y
submission.to_csv('s.csv', index=False)
submission.head(20)

