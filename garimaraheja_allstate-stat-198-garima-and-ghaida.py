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

#import plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt


# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings('ignore')

# Read raw data from the file

import pandas #provides data structures to quickly analyze data
#Since this code runs on Kaggle server, data can be accessed directly in the 'input' folder
#Read the train dataset
dataset = pandas.read_csv("../input/train.csv") 

#Read test dataset
dataset_test = pandas.read_csv("../input/test.csv")
#Save the id's for submission file
ID = dataset_test['id']
#Drop unnecessary columns
dataset_test.drop('id',axis=1,inplace=True)

#Print all rows and columns. Dont hide any
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

#Display the first five rows to get a feel of the data
print(dataset.head(5))

#Learning : cat1 to cat116 contain alphabets


#EXPLORATION STATISTICS: Find the skew of the distribution
print(dataset.skew())




#EXPLORATION GRAPHS: Box and Density plots for each variable to get a thorough understanding 

#range of features considered
split = 116 

#number of features considered
size = 15

#create a dataframe with only continuous features
data=dataset.iloc[:,split:] 

#get the names of all the columns
cols=data.columns 

#Plot violin for all attributes in a 7x2 grid
n_cols = 2
n_rows = 7

for i in range(n_rows):
    fg,ax = plt.subplots(nrows=1,ncols=n_cols,figsize=(12, 8))
    for j in range(n_cols):
        sns.violinplot(y=cols[i*n_cols+j], data=dataset, ax=ax[j])




#Skew Correction: we see that the skew is mostly corrected when the graph is displayed




#log1p function applies log(1+x) to all elements of the column
dataset["loss"] = numpy.log1p(dataset["loss"])
#visualize the transformed column
sns.violinplot(data=dataset,y="loss")  
plt.show()




#need to do




#get the number of rows and columns
r, c = dataset_encoded.shape

#create an array which has indexes of columns
i_cols = []
for i in range(0,c-1):
    i_cols.append(i)

#Y is the target column, X has the rest
X = dataset_encoded[:,0:(c-1)]
Y = dataset_encoded[:,(c-1)]
del dataset_encoded

#Validation chunk size
val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

#Split the data into chunks
from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
del X
del Y

#All features
X_all = []

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae = []

#Scoring parameter
from sklearn.metrics import mean_absolute_error

#Add this version of X to the list 
n = "All"
#X_all.append([n, X_train,X_val,i_cols])
X_all.append([n, i_cols])




#Evaluation of various combinations of LinearRegression

#Import the library
from sklearn.linear_model import LinearRegression

#uncomment the below lines if you want to run the algo
##Set the base model
#model = LinearRegression(n_jobs=-1)
#algo = "LR"
#
##Accuracy of the model using all features
#for name,i_cols_list in X_all:
#    model.fit(X_train[:,i_cols_list],Y_train)
#    result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
#    mae.append(result)
#    print(name + " %s" % result)
#comb.append(algo)

#Result obtained after running the algo. Comment the below two lines if you want to run the algo
mae.append(1278)
comb.append("LR" )    

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#MAE achieved is 1278




#Evaluation of various combinations of KNN

#Import the library
from sklearn.neighbors import KNeighborsRegressor

#Add the N value to the below list if you want to run the algo
n_list = numpy.array([])

for n_neighbors in n_list:
    #Set the base model
    model = KNeighborsRegressor(n_neighbors=n_neighbors,n_jobs=-1)
    
    algo = "KNN"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_neighbors )

if (len(n_list)==0):
    mae.append(1745)
    comb.append("KNN" + " %s" % 1 )
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Very high computation time
#Best estimated performance is 1745 for n=1




#Evaluation of various combinations of XGB

#Import the library
from xgboost import XGBRegressor

#Add the n_estimators value to the below list if you want to run the algo
n_list = numpy.array([])

for n_estimators in n_list:
    #Set the base model
    model = XGBRegressor(n_estimators=n_estimators,seed=seed)
    
    algo = "XGB"

    #Accuracy of the model using all features
    for name,i_cols_list in X_all:
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo + " %s" % n_estimators )

if (len(n_list)==0):
    mae.append(1169)
    comb.append("XGB" + " %s" % 1000 )    
    
##Set figure size
#plt.rc("figure", figsize=(25, 10))

##Plot the MAE of all combinations
#fig, ax = plt.subplots()
#plt.plot(mae)
##Set the tick names to names of combinations
#ax.set_xticks(range(len(comb)))
#ax.set_xticklabels(comb,rotation='vertical')
##Plot the accuracy for all combinations
#plt.show()    

#Best estimated performance is 1169 with n=1000# Make predictions using XGB as it gave the best estimated performance        

X = numpy.concatenate((X_train,X_val),axis=0)
del X_train
del X_val
Y = numpy.concatenate((Y_train,Y_val),axis=0)
del Y_train
del Y_val

n_estimators = 1000

#Best model definition
best_model = XGBRegressor(n_estimators=n_estimators,seed=seed)
best_model.fit(X,Y)
del X
del Y
#Read test dataset
dataset_test = pandas.read_csv("../input/test.csv")
#Drop unnecessary columns
ID = dataset_test['id']
dataset_test.drop('id',axis=1,inplace=True)

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset_test.iloc[:,i])
    feature = feature.reshape(dataset_test.shape[0], 1)
    #One hot encode
    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

del cats

#Concatenate encoded attributes with continuous attributes
X_test = numpy.concatenate((encoded_cats,dataset_test.iloc[:,split:].values),axis=1)

del encoded_cats
del dataset_test

#Make predictions using the best model
predictions = numpy.expm1(best_model.predict(X_test))
del X_test
# Write submissions to output file in the correct format
with open("submission.csv", "w") as subfile:
    subfile.write("id,loss\n")
    for i, pred in enumerate(list(predictions)):
        subfile.write("%s,%s\n"%(ID[i],pred))




#need to do
# 4 points for interpretation of any kind
# 3 points for thoroughness (did you fully grasp what your model is telling you about the data?)
# 3 points for comparing different models 



#need to do

# 7 points for correct out-of-sample test accuracy to the public leaderboard accuracy (within a reasonable threshold)  
# 3 points for demonstrating that your local test accuracy is correlated with the public leaderboard accuracy

