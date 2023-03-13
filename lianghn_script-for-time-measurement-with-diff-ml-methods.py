#!/usr/bin/env python
# coding: utf-8



# Load library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Class to predict the probability
class Class_Predict_test:
 def __init__(self, Number=100, Ncolumns=3): 
     self.Number=Number   # Number of Clients put in train sample
     self.Ncolumns=Ncolumns # Number of variables put in train sample

 def Predict_test(self,df_train,df_test,target):
# Prepraing Train Test Set        
    y = target[0:self.Number]
    X = df_train.iloc[:,0:self.Ncolumns].values[0:self.Number]

# Select Model
    clf = ExtraTreesClassifier(n_estimators=750,max_features=50,                                criterion= 'entropy',min_samples_split= 4,                                max_depth= 35, min_samples_leaf= 2,                                n_jobs = -1, random_state=12)
    
    start = time.time()
    clf.fit(X,y)
    end = time.time()
    totaltime = end - start
    print("clf.fit finished in {} seconds with {} clients      and {} variable".format(totaltime, self.Number, self.Ncolumns))

    # Prediction
    Test=df_test.iloc[:,0:self.Ncolumns].values[0:self.Number]
    Test_predict = clf.predict(Test)
    Test_proba=clf.predict_proba(Test)
    
    # Submission
    submission=pd.read_csv('../input/sample_submission.csv')
    submission=submission.iloc[0:self.Number,:]
    submission.index=submission.ID
    submission.PredictedProb=Test_proba[:,1]
    submission.to_csv('./BNP_Proba_ETC.csv', index=False)
    submission.PredictedProb.hist(bins=30)
    return;

# Load Data
df_train = pd.read_csv('../input/train.csv')     # 114321 rows x 133 columns
df_test = pd.read_csv('../input/test.csv')  # 114393 rows x 132 columns

# Drop columns
target = df_train['target'].values
df_train=df_train.drop(['ID','target'],axis=1)
df_test=df_test.drop(['ID'],axis=1)

# Feature Processing
refcols=df_train.columns
df_train=df_train.fillna(-999)
df_test=df_test.fillna(-999)

for elt in refcols:
    if df_train[elt].dtype=='O':
        df_train[elt], temp = pd.factorize(df_train[elt])
        df_test[elt]=temp.get_indexer(df_test[elt])
    else:
        df_train[elt]=df_train[elt].round(5)
        df_test[elt]=df_test[elt].round(5)
     




# Call the class and funtion to get prediction probability file.  
# Please play with different values of "Number" & "Ncolumns"
a = Class_Predict_test(Number=14000, Ncolumns=131)
a.Predict_test(df_train,df_test,target)






