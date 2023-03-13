#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')




train = pd.read_csv('../input/train.csv')
test_x = pd.read_csv('../input/test.csv')




train.head()




test_x.head()




print(train.isnull().any().any())
print(test_x.isnull().any().any())




train_y = train['y']
train_x = train
del train['y']




#Add column so we can easily split the data later
train_x['Source'] = 'train'
test_x['Source'] = 'test'

combined_x = train_x.append(test_x)




combined_x = pd.get_dummies(combined_x)
combined_x.head()




#Split back data
train_x = combined_x[combined_x['Source_train']==1]
del train_x['Source_train']
del train_x['Source_test']
test_x = combined_x[combined_x['Source_test']==1]
del test_x['Source_train']
del test_x['Source_test']




from sklearn import linear_model
from sklearn.cross_validation import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.33, random_state=42)




regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)




print("Intercept: %.2f" %regr.intercept_)
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_val) - y_val) ** 2))
# Explained variance score: 1 is perfect prediction
print('R-squared score: %.2f' % regr.score(X_val, y_val))




output = pd.DataFrame({'y': regr.predict(test_x)})
output['ID'] = test_x['ID']
output = output.set_index('ID')
output.to_csv('sub.csv')






