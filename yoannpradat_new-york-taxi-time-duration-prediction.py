#!/usr/bin/env python
# coding: utf-8



# -*- coding: utf-8 -*-
"""
Created on Sun Sep 03 22:33:34 2017

@author: YOANN
"""
    
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.tools.plotting import scatter_matrix


from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score




plt.close('all')

print("Loading and displaying data ... \n")
datafile = '../input/train.csv'
data = pd.read_csv(datafile)

m,n = data.shape

#=========================Having a look at the data =========================#
allLat = np.array(list(data['pickup_latitude'])+list(data['dropoff_latitude']))
allLong = np.array(list(data['pickup_longitude'])+list(data['dropoff_longitude']))

latLimits = [np.percentile(allLat,0.3), np.percentile(allLat, 99.7)]
lonLimits = [np.percentile(allLong,0.3), np.percentile(allLong, 99.7)]

durLimits  = [np.percentile(data['trip_duration'], 0.4), np.percentile(data['trip_duration'], 99.7)]

data = data[(data['pickup_latitude']>=latLimits[0])&(data['pickup_latitude']<=latLimits[1])]
data = data[(data['dropoff_latitude']>=latLimits[0])&(data['dropoff_latitude']<=latLimits[1])]
data = data[(data['pickup_longitude']>=lonLimits[0])&(data['pickup_longitude']<=lonLimits[1])]
data = data[(data['dropoff_longitude']>=lonLimits[0])&(data['dropoff_longitude']<=lonLimits[1])]
data = data[(data['trip_duration']>=durLimits[0])&(data['trip_duration']<=durLimits[1])]
    
allLat = np.array(list(data['pickup_latitude'])+list(data['dropoff_latitude']))
allLong = np.array(list(data['pickup_longitude'])+list(data['dropoff_longitude']))

medianLat  = np.percentile(allLat,50)
medianLong = np.percentile(allLong,50)

latMultiplier  = 111.32
longMultiplier = np.cos(medianLat*(np.pi/180.0)) * 111.32

data['duration [min]'] = data['trip_duration']/60.0
data['src lat [km]']   = latMultiplier  * (data['pickup_latitude']   - medianLat)
data['src long [km]']  = longMultiplier * (data['pickup_longitude']  - medianLong)
data['dst lat [km]']   = latMultiplier  * (data['dropoff_latitude']  - medianLat)
data['dst long [km]']  = longMultiplier * (data['dropoff_longitude'] - medianLong)

allLat  = np.array(list(data['src lat [km]'])  + list(data['dst lat [km]']))
allLong = np.array(list(data['src long [km]']) + list(data['dst long [km]']))


fig, axArray = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
axArray[0].set_ylabel('Count')
axArray[0].hist(allLat,80);axArray[1].set_xlabel('Lat in km')
axArray[1].hist(allLong,80);axArray[1].set_xlabel('Long in km')




# show the log density of pickup and dropoff locations
imageSize = (700,700)
longRange = [-5,19]
latRange = [-13,11]

allLatInds  = imageSize[0] - (imageSize[0] * (allLat  - latRange[0])  / (latRange[1]  - latRange[0]) ).astype(int)
allLongInds =                (imageSize[1] * (allLong - longRange[0]) / (longRange[1] - longRange[0])).astype(int)

locationDensityImage = np.zeros(imageSize)
for latInd, longInd in zip(allLatInds,allLongInds):
    locationDensityImage[latInd,longInd] += 1

fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(12,12))
ax.imshow(np.log(locationDensityImage+1),cmap='hot')
ax.set_axis_off()




pickUpTime = pd.to_datetime(data['pickup_datetime'])
data['src hourOfDay'] = pickUpTime.dt.hour + pickUpTime.dt.minute/60.0
data['day of week'] = pickUpTime.dt.weekday
data['month of year'] = pickUpTime.dt.month
    
delta_lat = np.radians(data['dropoff_latitude']-data['pickup_latitude'])
delta_lon = np.radians(data['dropoff_longitude']-data['pickup_longitude'])
lat_m = np.radians(data['dropoff_latitude']+data['pickup_latitude'])/2.0
    
data['distance'] = np.sqrt((data['dst lat [km]']-data['src lat [km]'])**2
                           +(data['dst long [km]']-data['src long [km]'])**2)

fig, axArray = plt.subplots(nrows=1,ncols=3,figsize=(12,5))
axArray[0].hist(data['duration [min]'],80)
axArray[0].set_xlabel('duration [min]'), axArray[0].set_ylabel('Count')
axArray[1].hist(data['distance'],80)
axArray[1].set_xlabel('distance in km')
axArray[2].scatter(data['distance'], data['duration [min]'])
axArray[2].set_xlabel('distance in km'); axArray[2].set_ylabel('trip duration in min')




data.head(1)




rand_rows = np.random.permutation(data.shape[0])
plt.scatter(data['distance'][rand_rows[0:100]], data['duration [min]'][rand_rows[0:100]])

X_tot = data.get(['distance','src lat [km]','src long [km]',
                         'dst lat [km]','dst long [km]', 'src hourOfDay', 
                         'day of week','month of year']).values
y_tot = data['trip_duration'].values
    
rand_rows = np.random.permutation(data.shape[0])
m_red = 200000
data_red = data.ix[rand_rows[0:m_red]].dropna()

X = data_red.get(['distance','src lat [km]','src long [km]',
                         'dst lat [km]','dst long [km]', 'src hourOfDay', 
                         'day of week','month of year']).values
y = data_red['trip_duration'].values

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = X/std
    
m_red = X.shape[0]
rand_rows = np.random.permutation(m_red)

m_train = np.int(0.5*m_red)
X_train = X[rand_rows[0:m_train],:]
y_train = y[rand_rows[0:m_train]]
X_test = X[rand_rows[m_train:],:]
y_test = y[rand_rows[m_train:]]




datafiletest = '../input/test.csv'
datatest = pd.read_csv(datafiletest)

pickUpTime = pd.to_datetime(data['pickup_datetime'])
pickUpTimeTest = pd.to_datetime(datatest['pickup_datetime'])
data['pickup_date'] = pickUpTime.dt.date
datatest['pickup_date'] = pickUpTimeTest.dt.date

plt.plot(data.groupby('pickup_date').count()[['id']], 'o-', label='train')
plt.plot(datatest.groupby('pickup_date').count()[['id']], 'o-', label='test')
plt.title('Train and test period complete overlap.')
plt.legend(loc=0)
plt.ylabel('number of records')
plt.show()




def RMSLE(estimator, X, y_true):
    y_pred = estimator.predict(X)
    n = y_pred.shape[0]
    
    for i in range(n):
        if (y_pred[i]<0):
            y_pred[i]=0
        
    return np.sqrt(sum((np.log(y_pred+1)-np.log(y_true+1))**2)/float(n))




#===========================Linear Regression==========================#
from sklearn.linear_model import Ridge

alpha_range = np.linspace(0.0001,10,10)

param_grid = dict(alpha=alpha_range)
cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 42)
grid = GridSearchCV(Ridge(), scoring = RMSLE, param_grid = param_grid, cv = cv)
grid.fit(X_train,y_train)

print("The best parameters are %s with a score of %0.5f"
      % (grid.best_params_, grid.best_score_))

print("Score for linear ridge is %.5f"%
      (RMSLE(grid.best_estimator_,X_test,y_test)))




from sklearn.neural_network import MLPRegressor

MLP_reg_clf = MLPRegressor(alpha=0.45, solver = 'lbfgs', hidden_layer_sizes=(5,4,3))
MLP_reg_clf.fit(X_train, y_train)

#alpha_range = np.linspace(0.001,0.5,5)

#param_grid = dict(alpha=alpha_range)
#cv = ShuffleSplit(n_splits = 5, test_size = 0.2, random_state = 42)
#grid = GridSearchCV(MLPRegressor(solver = 'lbfgs', hidden_layer_sizes=(5)), scoring = RMSLE, param_grid = param_grid, cv = cv)
#grid.fit(X,y)

#print("The best parameters are %s with a score of %0.5f"
#      % (grid.best_params_, grid.best_score_))

print("Score for MLP_reg is %.5f"%
      (RMSLE(MLP_reg_clf,X_test,y_test)))




from sklearn.cluster import MiniBatchKMeans

n_clusters=200
KMeans_clf = MiniBatchKMeans(n_clusters=n_clusters)
y_clusters_indexes = KMeans_clf.fit_predict(np.reshape(y_tot, (y_tot.shape[0],1)))
y_clusters_values = np.array([KMeans_clf.cluster_centers_[y_clusters_indexes[i]] 
                                for i in range(len(y_clusters_indexes))])

print("Comparison between original trip duration and clusters")
fig, axArray = plt.subplots(nrows=1,ncols=2,figsize=(8,5))
axArray[0].hist(y_tot,n_clusters)
axArray[0].set_xlabel('duration in sec');axArray[0].set_ylabel('Count')
axArray[1].hist(y_clusters_values,n_clusters);axArray[1].set_xlabel('duration in sec')






