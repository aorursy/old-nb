import numpy as np
import pandas as pd

data=pd.read_csv('../input/train.csv')
out_data=pd.read_csv('../input/test.csv')


x=data.loc[:,'var3':'var38']
y=data['TARGET']

x=np.asarray(x)
y=np.asarray(y)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import metrics
from sklearn.cross_validation import train_test_split

import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
training = training.replace(-999999,2)


# Replace 9999999999 with NaN
# See https://www.kaggle.com/c/santander-customer-satisfaction/forums/t/19291/data-dictionary/111360#post111360
# training = training.replace(9999999999, np.nan)
# training.dropna(inplace=True)
# Leads to validation_0-auc:0.839577

X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)
# # Add log of var38
# X['logvar38'] = X['var38'].map(np.log1p)
# # Encode var36 as category
# X['var36'] = X['var36'].astype('category')
# X = pd.get_dummies(X)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif,chi2
from sklearn.preprocessing import Binarizer, scale

p = 86 # 308 features validation_1-auc:0.848039
p = 80 # 284 features validation_1-auc:0.848414
p = 77 # 267 features validation_1-auc:0.848000
p = 75 # 261 features validation_1-auc:0.848642
# p = 73 # 257 features validation_1-auc:0.848338
# p = 70 # 259 features validation_1-auc:0.848588
# p = 69 # 238 features validation_1-auc:0.848547
# p = 67 # 247 features validation_1-auc:0.847925
# p = 65 # 240 features validation_1-auc:0.846769
# p = 60 # 222 features validation_1-auc:0.848581

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [ f for i,f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(chi2_selected.sum(),
   chi2_selected_features))
f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [ f for i,f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(f_classif_selected.sum(),
   f_classif_selected_features))
selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))
features = [ f for f,s in zip(X.columns, selected) if s]
print (features)


X_sel = X[features]

X_train, X_test, y_train, y_test = \
  cross_validation.train_test_split(X_sel, y, random_state=1301, stratify=y, test_size=0.3)

# xgboost parameter tuning with p = 75
# recipe: https://www.kaggle.com/c/bnp-paribas-cardif-claims-management/forums/t/19083/best-practices-for-parameter-tuning-on-models/108783#post108783

# max_depth=10, min_child_weight = 5 -> validation_1-auc:0.844981
# max_depth=9, learning_rate=0.1 -> validation_1-auc:0.840633
# max_depth=8 -> validation_1-auc:0.841643
# max_depth=7 -> validation_1-auc:0.841124
# max_depth=10 -> validation_1-auc:0.838350
# max_depth=8, subsample=0.6 -> validation_1-auc:0.838350
# max_depth=8, subsample=0.8 -> validation_1-auc:0.840091
# min_child_weight=6 -> validation_1-auc:0.842313
# min_child_weight=7 -> validation_1-auc:0.843404
# min_child_weight=8 -> validation_1-auc:0.841149
# min_child_weight=7, colsample_bytree=0.5 -> validation_1-auc:0.845604
# colsample_bytree=0.6 -> validation_1-auc:0.844602
# colsample_bytree=0.5, learning_rate=0.05 -> validation_1-auc:0.845504
# colsample_bytree=0.4 -> validation_1-auc:0.844258
# colsample_bytree=0.5, learning_rate=0.02 -> validation_1-auc:0.844924
# learning_rate=0.05 -> validation_1-auc:0.845504
# adding logvar38 -> validation_1-auc:0.845203
# removing logvar38, encode var36 as category -> validation_1-auc:0.835946
# remove extra features as they don't help

clf = xgb.XGBClassifier(missing=9999999999,
                max_depth = 8,
                n_estimators=1000,
                learning_rate=0.02, 
                nthread=4,
                subsample=0.8,
                colsample_bytree=0.5,
                min_child_weight = 7,
                seed=4242)
clf.fit(X_train, y_train, early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_train, y_train), (X_test, y_test)])
        
print('Overall AUC:', roc_auc_score(y, clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)[:,1]))

test['n0'] = (test == 0).sum(axis=1)
# test['logvar38'] = test['var38'].map(np.log1p)
# # Encode var36 as category
# test['var36'] = test['var36'].astype('category')
# test = pd.get_dummies(test)
sel_test = test[features]    
y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission.csv", index=False)

mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
ts = pd.Series(clf.booster().get_fscore())
#ts.index = ts.reset_index()['index'].map(mapFeat)
ts.sort_values()[-15:].plot(kind="barh", title=("features importance"))

featp = ts.sort_values()[-15:].plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
fig_featp = featp.get_figure()


import operator
sorted_x = sorted(a.items(), key=operator.itemgetter(1))

fe=['num_var22_hace3','num_med_var45_ult3','saldo_var5','saldo_var42','num_var45_hace3','num_var22_ult3','saldo_medio_var5_ult1','num_var45_ult3','saldo_medio_var5_hace2',
'saldo_var30',
'saldo_medio_var5_ult3',
'saldo_medio_var5_hace3',
'var15',
'var38']
x=data[fe]

gb=GradientBoostingClassifier()
fe=np.asarray(fe)
gb.fit(fe,y)


