#!/usr/bin/env python
# coding: utf-8




# coding: utf-8


import os
import pandas as pd
import numpy as np



file = pd.read_csv('../input/train.csv')

from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
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

file.head()



trip_duration = []
sum = 0
for duration in file['trip_duration']:
    trip_duration.append(duration)
    sum += duration



sum/len(trip_duration)



i=1
smallest = trip_duration[0]
largest = trip_duration[0]
while i<len(trip_duration):
    if trip_duration[i]<smallest:
        smallest = trip_duration[i]
    if trip_duration[i]>largest:
        largest = trip_duration[i]
    i += 1
print(smallest)
print(largest)







trip_duration.sort()



trip_duration.sort(reverse=True)



count = 0 
for duration in trip_duration :
    if duration<300 or duration>3600:
        count += 1



passenger_count_x=[]
trip_duration_y=[]
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
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
i=0
for index,row in file.iterrows():
    if i>50000:
        break
    if i<50000:
        p_s = []
        if row[-1]>300 and row['trip_duration']<3600:
            p_s.append(row['passenger_count'])
            p_s.append(haversine(row['pickup_longitude'],row['pickup_latitude'],row['dropoff_longitude'],row['dropoff_latitude']))
            p_s.append(int(row['pickup_datetime'][-8:-6]))
            passenger_count_x.append(p_s)
            trip_duration_y.append(row['trip_duration'])
    i += 1



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(passenger_count_x,trip_duration_y)



from sklearn.neighbors import KNeighborsRegressor



knn = KNeighborsRegressor(n_neighbors=10,algorithm='ball_tree',n_jobs=-1)



y_predicted = knn.fit(x_train,y_train).predict(x_test)



from sklearn.metrics import accuracy_score



y_test = np.array(y_test)



from sklearn.metrics import mean_squared_error



def rmsle(y_predicted,y_test):
    sum=0.0
    for x in range(len(y_predicted)):
        p = np.log(y_predicted[x]+1)
        r = np.log(y_test[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(y_predicted))**0.5



rmsle(y_predicted,y_test)



from sklearn.neighbors import kneighbors_graph



kneighbors_graph(x_train,n_neighbors=2)










