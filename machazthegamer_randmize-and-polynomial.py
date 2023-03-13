#!/usr/bin/env python
# coding: utf-8



"""
drop all columns with nans
look for the remainng 100 least correlated variables with price
get some polynomial features on them i.e square them
merge the polynomials with the other more correlated features
do randomised lasso to find right features
do GTR on them
"""




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




df = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
price = df.price_doc
df_train= df.drop(labels=["price_doc"], axis=1)




df_combine = pd.concat([df_train,df_test], ignore_index=True, axis=0).dropna(axis=1, how="any")


obj_col = df_combine.select_dtypes(include=[object]).columns
from sklearn.preprocessing import LabelEncoder
for name in obj_col:
    if name != "timestamp" and name != "product_type":
        print(name)
        encoder = LabelEncoder()
        df_combine[name] = encoder.fit_transform(df_combine[name].values)
        




df_test = df_combine.iloc[30471:, :].drop(labels=["timestamp"], axis=1)
df_train = df_combine.iloc[0:30471, :].drop(labels=["timestamp"], axis=1)




df_test = df_test.drop(labels="id", axis=1)
df_train = df_train.drop(labels="id", axis=1)




corrs = df_train.corrwith(price)

to_square = corrs.abs().sort_values(ascending=False).tail(100).index

df_train[to_square] = df_train[to_square]**2
df_test[to_square] =  df_test[to_square]**2

from sklearn.linear_model import RandomizedLasso
clf = RandomizedLasso(n_resampling=200)
clf.fit(X=df_train, y=price.values)




pd.set_option("display.max_rows", 20)
np.set_printoptions(threshold=np.nan)
important = df_train.columns[clf.get_support()]

modified_test = clf.transform(X=df_test.values)
modified_train = clf.transform(X=df_train.values)

from sklearn.ensemble import GradientBoostingRegressor
reg = GradientBoostingRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=10)
reg.fit(X=modified_train, y=price.values)
reg.score(X=modified_train, y=price.values)




predictions = reg.predict(X=modified_test)
submission = pd.Series(index=df_combine.id[30471:].values, data=predictions, name="price_doc")
submission.index.name="id"




important_polys = set(to_square) & set(important)
print("thre are {} important polys from the original {}".format(len(important_polys), len(to_square)))




pd.set_option("display.max_rows", 20)
feature_importance = pd.Series(data=reg.feature_importances_, index=important).sort_values(ascending=False)
print("non_zero_coeffs: {}\n all_coefficients: {}".format(feature_importance[feature_importance>0].shape, feature_importance.shape))




feature_importance[feature_importance>0][important_polys].sort_values(ascending=False)




feature_importance[feature_importance>0]




get_ipython().run_line_magic('matplotlib', 'inline')
train_predict = reg.predict(X=modified_train)
import matplotlib.pyplot as plt
dates = pd.to_datetime(df_combine["timestamp"].values)
train_dates = dates[:30471]




plt.rcParams["figure.figsize"] = 20,20
_, _, hist = plt.hist(price.values, 100, facecolor="red", alpha=0.6)
_, _, hist = plt.hist(train_predict, 100, facecolor="green", alpha=0.5)
plt.label=["actual price", "predicted"]




data={"train_dates":train_dates.values, "train_predict":train_predict, "price":price.values}




data_f = pd.DataFrame(data=data)
data_f = data_f.assign(month=data_f.train_dates.dt.month).assign(year=data_f.train_dates.dt.year)
data_f = data_f.assign(monthyear=data_f.year.astype(str)+data_f["month"].astype(str)).sort_values("year")




plt.rcParams["figure.figsize"] = 20,20
data_f.drop(labels=["train_dates", "month", "year"], axis=1).groupby(by="monthyear").mean().plot(kind="line")




plt.rcParams["figure.figsize"] = 30,30
data_f.assign(sub_area=df_combine.sub_area).assign(price_diff=data_f.train_predict-data_f.price).drop(labels=["train_dates", "month", "year", "train_predict", "price"], axis=1).groupby(by="sub_area").mean()[:52].sort_values("price_diff").plot(kind="bar")




df_macro = pd.read_csv("../input/macro.csv")




df_macro.dropna(axis=1, how="any").columns






