#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
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




import matplotlib.pyplot as plt
import seaborn as sns




from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer




df = pd.read_csv('../input/train.csv')




df.columns




soiltype_df = df[[i for i in df.columns if 'soil_type' in i.lower()]]




df.head()




plt.close('all')
g = df.groupby('Cover_Type')





afig, ax = plt.subplots(figsize=(10, 7))
h = ax.scatter(df['Elevation'], df['Slope'], c=df['Cover_Type'],
               alpha=0.5, cmap='viridis')
fig.colorbar(h, ax=ax)




#names of all the columns
cols = df.columns

#number of rows=r , number of columns=c
r,c = df.shape

#Create a new dataframe with r rows, one column for each encoded category,
# and target in the end
data = pd.DataFrame(index=np.arange(0, r),
                    columns=['Wilderness_Area','Soil_Type','Cover_Type'])

#Make an entry in 'data' for each r as category_id, target value
for i in range(0,r):
    w = 0
    s = 0
    # Category1 range
    for j in range(10,14):
        if (df.iloc[i, j] == 1):
            w=j-9  #category class
            break
    # Category2 range        
    for k in range(14,54):
        if (df.iloc[i,k] == 1):
            s=k-13 #category class
            break
    #Make an entry in 'data' for each r as category_id, target value        
    data.iloc[i]=[w, s, df.iloc[i,c-1]]




fig = plt.figure(figsize=(32, 6))
sns.countplot(figure=fig, x="Soil_Type", hue="Cover_Type", data=data)




df.groupby('Cover_Type').size()




df.describe()




from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer




#Removal list initialize
rem = []

#Add constant columns as they don't help in prediction process
for c in df.columns:
    if df[c].std() == 0: #standard deviation is zero
        rem.append(c)

#drop the columns        
new_df = df.drop(rem,axis=1)

print(rem)




#get the number of rows and columns
r, c = df.shape

#get the list of columns
cols = df.columns
#create an array which has indexes of columns
i_cols = list(range(c-1))

#array of importance rank of all features  
ranks = []

#Extract only the values
array = df.values

#Y is the target column, X has the rest
X_orig = array[:, :(c-1)].copy()
Y = array[:, c-1].copy()

#Validation chunk size
val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0




X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X_orig, Y,
                                                                   test_size=val_size,
                                                                   random_state=seed)




X_train






