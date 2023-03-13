# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# www.kaggle.com/c/4986/scripts/notebook

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import xgboost as xgb
import sklearn
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

# Read data
df_train = pd.read_csv("../input/train.csv", index_col='ID')
feature_cols = list(df_train.columns)
feature_cols.remove("TARGET")
df_test = pd.read_csv("../input/test.csv", index_col='ID')

# Split up the data
X_all = df_train[feature_cols]
y_all = df_train["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=5, stratify=y_all)

# Get top features from xgb model
model = xgb.XGBRegressor(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=9,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=5
)

# Train cv
xgb_param = model.get_xgb_params()
dtrain = xgb.DMatrix(X_train.values, label=y_train.values, missing=np.nan)
cv_result = xgb.cv(
    xgb_param,
    dtrain,
    num_boost_round=model.get_params()['n_estimators'],
    nfold=5,
    metrics=['auc'],
    early_stopping_rounds=50)
best_n_estimators = cv_result.shape[0]
model.set_params(n_estimators=best_n_estimators)

# Train model
model.fit(X_train, y_train, eval_metric='auc')

# Predict training data
y_hat_train = model.predict(X_train)

# Predict test data
y_hat_test = model.predict(X_test)

# Print model report:
print("\nModel Report")
print("best n_estimators: {}".format(best_n_estimators))
print("AUC Score (Train): %f" % roc_auc_score(y_train, y_hat_train))
print("AUC Score (Test) : %f" % roc_auc_score(y_test,  y_hat_test))

# Get important features
feat_imp = list(pd.Series(model.booster().get_fscore()).sort_values(ascending=False).index)

# Even out the targets
df_train_1 = df_train[df_train["TARGET"] == 1]
df_train_0 = df_train[df_train["TARGET"] == 0].head(df_train_1.shape[0])
df_train = df_train_1.append(df_train_0)

# Scale data
X_all = df_train[feat_imp].copy(deep=True)
y_all = df_train["TARGET"]
X_all[feat_imp] = sklearn.preprocessing.scale(X_all, axis=0, with_mean=True, with_std=True, copy=True)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.3, random_state=5, stratify=y_all)

# Create second complementary column at position 0
y_train_2cols = np.array(list(zip((1 - y_train).values, y_train.values)))


