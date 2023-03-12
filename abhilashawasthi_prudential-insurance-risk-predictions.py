# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb
# get prudential & test csv files as a DataFrame
prudential_df  = pd.read_csv('../input/train.csv')
test_df        = pd.read_csv('../input/test.csv')

# preview the data
prudential_df.head()
prudential_df.info()
print("----------------------------")
test_df.info()
# response

fig, (axis1) = plt.subplots(1,1,figsize=(15,5))

sns.countplot(x=prudential_df["Response"], order=[1,2,3,4,5,6,7,8], ax=axis1)
# There are some columns with non-numerical values(i.e. dtype='object'),
# So, We will create a corresponding unique numerical value for each non-numerical value in a column of training and testing set.

from sklearn import preprocessing

for f in prudential_df.columns:
    if prudential_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(prudential_df[f].values) + list(test_df[f].values)))
        prudential_df[f] = lbl.transform(list(prudential_df[f].values))
        test_df[f]       = lbl.transform(list(test_df[f].values))
# fill NaN values

for f in prudential_df.columns:
    if f == "Response": continue
    if prudential_df[f].dtype == 'float64':
        prudential_df[f].fillna(prudential_df[f].mean(), inplace=True)
        test_df[f].fillna(test_df[f].mean(), inplace=True)
    else:
        prudential_df[f].fillna(prudential_df[f].median(), inplace=True)
        test_df[f].fillna(test_df[f].median(), inplace=True)

# prudential_df.fillna(0, inplace=True)
# test_df.fillna(0, inplace=True)
# define training and testing sets

X_train = prudential_df.drop(["Response", "Id"],axis=1)
Y_train = prudential_df["Response"]
X_test  = test_df.drop("Id",axis=1).copy()
#ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=100)
etc.fit(X_train,Y_train)
Y_test = etc.predict(X_test)
etc.score(X_train,Y_train)
Y_test = Y_test.astype(int)
# Create submission

submission = pd.DataFrame({
        "Id": test_df["Id"],
        "Response": Y_test
    })
submission.to_csv('etc.csv', index=False)
