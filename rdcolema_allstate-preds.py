#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV, BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings

warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')




### load data

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
full_df = pd.concat([train, test])

train.head()




### remove outliers (maybe)

print train.shape

train = train[train.loss < 25000]

print train.shape




### convert loss to log of loss (maybe)

# train["loss"] = np.log(train["loss"])




### feature encoding and normalization

cat_cols = [c for c in df.columns if "cat" in c]
cont_cols = [c for c in df.columns if "cont" in c]
all_cols = cat_cols + cont_cols


le = LabelEncoder()

for col in cat_cols:
    le.fit(df[col].values)
    train[col] = le.transform(train[col].values)
    test[col] = le.transform(test[col].values)

    
sc = StandardScaler()     # alt options: MinMaxScaler, RobustScaler, or no normalization

for col in all_cols:
    sc.fit(train[col].values)
    train[col] = sc.transform(train[col].values)
    test[col] = sc.transform(test[col].values)
    
train.head()












































### visualize spread between predictions and target values 

pred = np.zeros_like(y)
pred[:y.shape[0]] = [x for x in model.predict(X)]
# pred[:y.shape[0]] = ensemble_regressor(regressors, X)

fig, ax = plt.subplots()
ax.scatter(y, pred, c='k')
ax.plot([21000, 0], [21000, 0], 'r-', lw=2)
ax.set_xlabel('Actual value')
ax.set_ylabel('Predicted value')
plt.show()




### batch prediction

ids = test["id"].values

# predictions = ensemble_regressor(regressors, test.drop(["id"], axis=1).as_matrix())
predictions = ensemble_regressor([model], test.drop(["id"], axis=1).as_matrix())

# # w/ log loss
# with open("prediction.csv", "w") as f:
#     p_writer = csv.writer(f, delimiter=',', lineterminator='\n')
#     for i, p in enumerate(predictions):
#         p_writer.writerow([ids[i], np.exp(p)])
        
# w/o log loss
with open("prediction.csv", "w") as f:
    p_writer = csv.writer(f, delimiter=',', lineterminator='\n')
    for i, p in enumerate(predictions):
        p_writer.writerow([ids[i], p])

