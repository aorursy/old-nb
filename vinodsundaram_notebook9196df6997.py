#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")




data=train
print("Dimension of the dataset:",data.shape)

print(data.describe()) ## for all continuous variable
## we find that all cont variable are between 0-1 probabilities

## Mix of categorical and continous variables. Better to identify and split
col_cat=[col for col in data.columns if 'cat' in col]
col_cont=[col for col in data.columns if 'cont' in col]

#print("categorical column size: ", len(col_cat), " Continuous column size : ", len(col_cont))




fig=plt.figure(figsize=(8,3))
ax1 = fig.add_subplot(121)
data['cont1'].plot(kind='density',xlim=(0,1),title="cont1")
ax2 = fig.add_subplot(122)
data['cont2'].plot(kind='density',xlim=(0,1))




## need to convert categorical variables into numeric - cat1 to cat116
categorical_split=116
data_encoded=data
## LE or one hot. first trying LE
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in range(0,categorical_split):
        #data[i]=le.fit(data[i])
        print(data[i,"loss"])




data.dftypes()




dtypes(data)

