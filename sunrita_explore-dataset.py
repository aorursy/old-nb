#!/usr/bin/env python
# coding: utf-8



from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import model_selection, preprocessing, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss# This Python 3 environment comes with many helpful analytics libraries installed
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
color = sns.color_palette()




train_df = pd.read_json("../input/train.json")
train_df.head()




#ulimit = np.percentile(train_df.price.values, 90)
#llimit = np.percentile(train_df.price.values, 0)
#train_df['price'].ix[train_df['price']>ulimit] = ulimit
#new_train = new_train.ix[new_train['price']>llimit]

#int_level = new_train['interest_level'].value_counts()
#train_df['price_level'] = np.round(train_df['price']*10/ulimit)
#plt.figure(figsize=(8,4))
#sns.barplot(int_level.index, int_level.values, alpha=0.8, color=color[1])
#sns.countplot(x='interest_level', hue = 'price_level', data = train_df)
#plt.ylabel('Number of Occurrences', fontsize=12)
#plt.xlabel('Interest level', fontsize=12)
#plt.show()

ulimit = np.percentile(train_df.price.values, 95)
train_df['price'].ix[train_df['price']>ulimit] = ulimit

train_df["num_photos"] = train_df["photos"].apply(len)
train_df["num_features"] = train_df["features"].apply(len)
train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
train_df["created"] = pd.to_datetime(train_df["created"])
#train_df["created_year"] = train_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
train_df = train_df[train_df["latitude"] != 0]
train_df = train_df[train_df["longitude"] != 0]

num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words", "created_month", "created_day"]




categorical = ["display_address", "manager_id", "building_id", "street_address"]
for f in categorical:
        if train_df[f].dtype=='object':
            #print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            num_feats.append(f)




X = train_df[num_feats]
y = train_df["interest_level"]
X.head()




X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)




clf = RandomForestClassifier(n_estimators=2000)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
log_loss(y_val, y_val_pred)




feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
pvals = feature_importance[sorted_idx]
pcols = X_train.columns[sorted_idx]
plt.figure(figsize=(8,12))
plt.barh(pos, pvals, align='center')
plt.yticks(pos, pcols)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')




#clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
#clf.fit(X_train, y_train)
#y_val_pred = clf.predict_proba(X_val)
#log_loss(y_val, y_val_pred)




#clf = AdaBoostClassifier(n_estimators=2000)
#clf.fit(X_train, y_train)
#y_val_pred = clf.predict_proba(X_val)
#log_loss(y_val, y_val_pred)




clf = AdaBoostClassifier(n_estimators=2000)
clf.fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
log_loss(y_val, y_val_pred)




feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
pvals = feature_importance[sorted_idx]
pcols = X_train.columns[sorted_idx]
plt.figure(figsize=(8,12))
plt.barh(pos, pvals, align='center')
plt.yticks(pos, pcols)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')




clf = GradientBoostingClassifier(n_estimators=2000, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X_train, y_train)
y_val_pred = clf.predict_proba(X_val)
log_loss(y_val, y_val_pred)




feature_importance = clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
pvals = feature_importance[sorted_idx]
pcols = X_train.columns[sorted_idx]
plt.figure(figsize=(8,12))
plt.barh(pos, pvals, align='center')
plt.yticks(pos, pcols)
plt.xlabel('Relative Importance')
plt.title('Variable Importance')

