#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




df = pd.read_csv("../input/train.csv")
df.head()




corr = df.corr()
for a in corr.columns:
    for b in corr.columns:
        if abs(corr[a][b]) > .8 and a > b:
            print(corr[a][b])
            df.plot(kind='scatter', x=a, y=b)
            plt.show()




from sklearn import cross_validation
def prepare(df, dropped = ['id']):
    for c in df.columns:
        if len(df[c].unique()) == 2:
            el = df[c][0]
            df[c] = df[c].apply(lambda x: x == el)
    dropped += [c for c in df.columns if df[c].dtype.kind not in 'biufc' or max(df.groupby(c).size())/float(len(df)) > .98]
    return df.drop(dropped, axis = 1)

def prepareTrain(df):
    df = df.dropna()
    X, Y = df.drop('loss', axis = 1), df['loss']
    val_size = 0.1
    seed = 0
    return cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)




df = prepare(df)




df.head()




from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from keras.wrappers.scikit_learn import KerasRegressor
import numpy

def knn(n):
    X_train, X_val, Y_train, Y_val = prepareTrain(df)
    model = KNeighborsRegressor(n_neighbors=n,n_jobs=-1)
    algo = "KNN"
    model.fit(X_train, Y_train)
    return mean_absolute_error(model.predict(X_val), Y_val)
print(knn(1))




mean_absolute_error(m.predict(X_val))

