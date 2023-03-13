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




###### load the data
data_train=pd.read_csv("../input/train.csv",index_col=0)
data_test=pd.read_csv("../input/train.csv",index_col=0)




from sklearn.preprocessing import PolynomialFeatures





data_train.head()




##########seperate the data into feature and label
train_X=data_train.iloc[:,0:5]
train_y=data_train.iloc[:,5]




train_X_number=train_X.iloc[:,0:4]






