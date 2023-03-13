#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt    #--- for plotting ---
import numpy as np                 #--- linear algebra ---
import pandas as pd                #--- data processing, CSV file I/O (e.g. pd.read_csv) ---
import seaborn as sns              #--- for plotting and visualizations ---

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Input data files are available in the "../input/" directory.
path = 'D:/BACKUP/Kaggle/New York City Taxi/Data/'
train_df = pd.read_csv('../input/train.csv')

#--- Let's peek into the data
print (train_df.head())




from math import radians, cos, sin, asin, sqrt   #--- for the mathematical operations involved in the function ---

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

train_df['Displacement (km)'] = train_df.apply(lambda x: haversine(x['pickup_longitude'], x['pickup_latitude'], x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
#train_df['Haversine_dist'] = haversine(train_df['pickup_longitude'], train_df['pickup_latitude'],train_df['dropoff_longitude'], train_df['dropoff_latitude'])
print (train_df.head())




train_df = train_df.rename(columns = {'Displacement (km)' : 'Haversine_dist'})
#df=df.rename(columns = {'two':'new_name'})
print (train_df.head())




def arrays_bearing(lats1, lngs1, lats2, lngs2, R=6371):
    lats1_rads = np.radians(lats1)
    lats2_rads = np.radians(lats2)
    lngs1_rads = np.radians(lngs1)
    lngs2_rads = np.radians(lngs2)
    lngs_delta_rads = np.radians(lngs2 - lngs1)
    
    y = np.sin(lngs_delta_rads) * np.cos(lats2_rads)
    x = np.cos(lats1_rads) * np.sin(lats2_rads) - np.sin(lats1_rads) * np.cos(lats2_rads) * np.cos(lngs_delta_rads)
    
    return np.degrees(np.arctan2(y, x))

train_df['bearing_dist'] = arrays_bearing(
train_df['pickup_latitude'], train_df['pickup_longitude'], 
train_df['dropoff_latitude'], train_df['dropoff_longitude'])

print (train_df.head())




#--- Taken from Part 2 ---
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'])
train_df['dropoff_datetime'] = pd.to_datetime(train_df['dropoff_datetime'])

train_df['pickup_month'] = train_df.pickup_datetime.dt.month.astype(np.uint8)
train_df['pickup_day'] = train_df.pickup_datetime.dt.weekday.astype(np.uint8)
train_df['pickup_hour'] = train_df.pickup_datetime.dt.hour.astype(np.uint8)

train_df['dropoff_month'] = train_df.dropoff_datetime.dt.month.astype(np.uint8)
train_df['dropoff_day'] = train_df.dropoff_datetime.dt.weekday.astype(np.uint8)
train_df['dropoff_hour'] = train_df.dropoff_datetime.dt.hour.astype(np.uint8)
print (train_df.head())




train_df['Manhattan_dist'] =     (train_df['dropoff_longitude'] - train_df['pickup_longitude']).abs() +     (train_df['dropoff_latitude'] - train_df['pickup_latitude']).abs()
    
print(train_df.head())    





print('Range of Haversine_dist is {:f} to {:f}'.format(max(train_df['Haversine_dist']),min(train_df['Haversine_dist'])))
print('Range of Manhattan_dist is {:f} to {:f}'.format(max(train_df['Manhattan_dist']),min(train_df['Manhattan_dist'])))
print('Range of Bearing_dist is {:f} to {:f}'.format(max(train_df['bearing_dist']),min(train_df['bearing_dist'])))

  




#--- get the distance metrics in a df ---
distance_df = train_df[['Haversine_dist','bearing_dist','Manhattan_dist']]
print (distance_df.corr())




data = train_df.groupby('pickup_month').aggregate({'Haversine_dist':'sum'}).reset_index()
sns.barplot(x='pickup_month', y='Haversine_dist', data=data)
plt.title('Pick-up Month vs Haversine_dist')
plt.xlabel('Pick-up Month')
months = ['January', 'February', 'March', 'April', 'May', 'June']
plt.xticks(range(0,7), months, rotation='horizontal')
plt.ylabel('Displacement (km)')




data = train_df.groupby('pickup_day').aggregate({'Haversine_dist':'sum'}).reset_index()
sns.barplot(x='pickup_day', y='Haversine_dist', data=data)
plt.title('Pick-up Day vs Haversine_dist')
plt.xlabel('Pick-up Month')
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.xticks(range(0,8), days, rotation='horizontal')
plt.ylabel('Displacement (km)')




data = train_df.groupby('pickup_hour').aggregate({'Haversine_dist':'sum'}).reset_index()
sns.barplot(x='pickup_hour', y='Haversine_dist', data=data)
plt.title('Pick-up Hour vs Haversine_dist')
plt.xlabel('Pick-up Hour')
#plt.xticks(range(0,8), days, rotation='horizontal')
plt.ylabel('Displacement (km)')




data = train_df.groupby('dropoff_hour').aggregate({'Haversine_dist':'sum'}).reset_index()
sns.barplot(x='dropoff_hour', y='Haversine_dist', data=data)
plt.title('Drop-off Hour vs Haversine_dist')
plt.xlabel('Drop-off Hour')
#plt.xticks(range(0,8), days, rotation='horizontal')
plt.ylabel('Displacement (km)')

