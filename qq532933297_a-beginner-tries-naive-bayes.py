import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn.preprocessing as preprocessing


#　Load data
Data = pd.read_csv('../input/train.csv')
Data = Data.loc[Data.shape[0]*0.8:,:] # use the latest data

# ----------------------------------Rubbish bin--------------------------------------------
#mapData = np.loadtxt('../input/sf_map_copyright_openstreetmap_contributors.txt')
#fig = plt.figure(figsize = (11.69, 8.27))
#plt.imshow(mapData, cmap=plt.get_cmap('gray'),extent=lon_lat_box)

#ax = plt.figure(figsize=(14,8))
#trainData.DayOfWeek.value_counts().plot(kind='bar')
#plt.title('Crime occuring in each day')
#plt.ylabel('Number') 

#ax = plt.figure(figsize=(14,8))
#trainData[trainData.Category == 'LARCENY/THEFT'].Dates.value_counts().plot(kind='bar')
#plt.title('Crime occuring in each day')
#plt.ylabel('Number')
# --------------------------------------------------------------------
# Get hours in each day

#result = np.zeros((validationY.shape[0],validationY.shape[1]))
# learning
#for i in range(20):


# Change the format of Dates
Data.Dates = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in Data.Dates])
#Data['Time'] = np.array([date.hour for date in Data.Dates])
Data['Time'] = np.array([np.where(date.hour>11,1,0) for date in Data.Dates])



# Take samples 
sample_x = pd.DataFrame([])
sample_y = pd.DataFrame([])
    # factorization
dummies_PdDistrict = pd.get_dummies(Data['PdDistrict'], prefix= 'PdDistrict')
sample_x= pd.concat([sample_x, dummies_PdDistrict], axis=1)
          #DayOfWeek seems irrelevant to the crimes
#dummies_DayOfWeek = pd.get_dummies(Data['DayOfWeek'], prefix= 'DayOfWeek')      
#sample_x = pd.concat([sample_x, dummies_DayOfWeek], axis=1)  
dummies_Category = pd.get_dummies(Data['Category'], prefix= 'Category')
sample_y = pd.concat([sample_y, dummies_Category], axis=1)   


C = 'LARCENY/THEFT'
sample_x['distanceToCenterX'] = Data.X-Data[Data.Category == C].X.mean()
sample_x['distanceToCenterY'] = Data.Y-Data[Data.Category == C].Y.mean()

#ax = plt.figure(figsize=(14,8))
#Data[Data.Category==C].Time.value_counts().plot(kind='bar')
#plt.title(' What time did the cirme happen')
#plt.ylabel('Number') 



from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB

trainX, cvX = cross_validation.train_test_split(x, test_size=0.3, random_state=0)
trainY, cvY = cross_validation.train_test_split(y, test_size=0.3, random_state=0)

#clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf = GaussianNB()
clf.fit(trainX.as_matrix(),trainY.as_matrix())
prob = clf.predict_proba(cvX)



pdt = pd.DataFrame([np.where(prob[i,0]<0.5,1,0) for i in range(prob.shape[0])],columns = ['Predicted'])
bad_pdt = cvX[pdt['Predicted'] != cvY]

print(cvY)
print(pdt)
