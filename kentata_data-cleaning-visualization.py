#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt




train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")




print(train.shape)
train.head()




print(test.shape)
test.head()




len(np.unique(train['id']))




len(np.unique(train['num_room']) )




train = train.set_index('timestamp')
test = test.set_index("timestamp")
train.head()




train.drop("id", axis=1, inplace=True)




train.head()




train.describe()




train.shape[0]




train['price_doc'].values




ax = train['price_doc'].plot(style=['-'])
ax.lines[0].set_alpha(0.8)
plt.xticks(rotation=90)
plt.title("linear scale")
ax.legend()




ax = train['price_doc'].plot(style=['-'])
ax.lines[0].set_alpha(0.3)
ax.set_yscale('log')
plt.xticks(rotation=90)
plt.title("logarithmic scale")
ax.legend()




# no missing value for the target value
train[train['price_doc'].isnull()]




train.columns[train.isnull().any()]




train2 = train.fillna(train.median())




train2.head()




train2.columns[train2.isnull().any()]




train2.corr()




categorical = []
for i in train2.columns:
    if type(train2[i].values[0]) == str:
        categorical.append(i)
print(categorical)
print(len(categorical))




train2.shape




train2[categorical].head()




np.unique(train2['product_type'])




for cat in categorical:
    print(cat, ':', np.unique(train2[cat]))




yes_no_mapping = {'no': 0, 'yes': 1}




# ordinal features which could be rendered as 0 and 1,
# each corresponding to 'no' and 'yes'
categorical[2:-1]




for i in categorical[2:-1]:
    train2[i] = train2[i].map(yes_no_mapping)




categorical = []
for i in train2.columns:
    if type(train2[i].values[0]) == str:
        categorical.append(i)
print(categorical)
print(len(categorical))




np.unique(train2['ecology'].values)




rate_mapping = {'excellent': 3, 'good': 2, 'satisfactory': 2, 'poor': 1, 'no data': np.nan} 




train2['ecology'] = train2['ecology'].map(rate_mapping)




print(len(train2[train2['ecology'].isnull()]))




print(len(train2[train2['ecology'].notnull()]))




print(train2.shape[0])




print(len(train2[train2['ecology'].isnull()]) + len(train2[train2['ecology'].notnull()]))




train2 = train2.fillna(train2.median())




print(len(train2[train2['ecology'].isnull()]))




train2.corr()





train2.head()





train2.head() 



test = pd.read_csv("test.csv")




test.head()




test = test.set_index('timestamp')
test.head()




test.drop("id", axis=1, inplace=True)
print(test.shape)




for i in test.columns:
    if i not in train.columns:
        print(i)




categorical = []
for i in test.columns:
    if type(test[i].values[0]) == str:
        categorical.append(i)
print(categorical)
print(len(categorical))




categorical[2:-1]




for i in categorical[2:-1]:
    test[i] = test[i].map(yes_no_mapping)




test['ecology'] = test['ecology'].map(rate_mapping)




len(test[test['ecology'].isnull()])




test = test.fillna(test.median())




test.columns[test.isnull().any()]




# there are 33 missing values in a column called 'producty_type'
len(test[test['product_type'].isnull()])

train2.head()






