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




from sklearn import datasets




iris = datasets.load_iris()




digits = datasets.load_digits()




print(digits.data)  




digits.target




digits.images[0]




from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)




clf.fit(digits.data[:-1], digits.target[:-1])




clf.predict(digits.data[-1:])




import pickle




s = pickle.dumps(clf)




clf2 = pickle.loads(s)




clf2.predict(digits.data[-2:])




from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)




from sklearn.datasets import dump_svmlight_file
dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')




param = {
'max_depth': 3, # the maximum depth of each tree
'eta': 0.3, # the training step for each iteration
'silent': 1, # logging mode - quiet
'objective': 'multi:softprob', # error evaluation for multiclass training
'num_class': 3} # the number of classes that exist in this datset
num_round = 20 # the number of training iterations




bst = xgb.train(param, dtrain, num_round)




preds = bst.predict(dtest)




import numpy as np
best_preds = np.asarray([np.argmax(line) for line in preds])




from sklearn.metrics import precision_score
print (precision_score(y_test, best_preds, average='macro'))




from sklearn.externals import joblib
joblib.dump(bst, 'bst_model.pkl', compress=True)
# bst = joblib.load('bst_model.pkl') # load it later

