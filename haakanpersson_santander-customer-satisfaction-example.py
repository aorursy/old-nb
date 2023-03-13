#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt #plotting

import seaborn as sns #plotting

import xgboost as xgb #Gradient boosting
from sklearn.model_selection import train_test_split #model selection
from sklearn.feature_selection import SelectFromModel #model selection
from sklearn.metrics import accuracy_score #model testing




# Read data
train_df = pd.read_csv("../input/train.csv")
train_df.set_index("ID", verify_integrity = True, inplace = True)
print("No of features:")
no_of_features = train_df.shape[1]
print(no_of_features)
test_df = pd.read_csv("../input/test.csv")
test_df.set_index("ID",verify_integrity = True, inplace = True)
print('_'*40)
# Inspect training data
print(train_df.head())

print('_'*40)
print("Number of unsatisfied customers:", train_df["TARGET"].sum()) #  % train["TARGET"].sum()
print("Total number of customers:",  train_df["TARGET"].count())
print("Percentage of unsatisfied customers:", train_df["TARGET"].sum() / train_df["TARGET"].count())




# Examining the variance of the features
std = train_df.std()
print((std.sort_values(ascending = False)))

# Dropping features with zero variance
criterium = (std == 0)
train_df.drop(criterium.index[criterium], axis = 1, inplace = True)
(no_of_features_old, no_of_features) = (no_of_features, train_df.shape[1])
print("%d features with constant entries dropped." % (no_of_features_old - no_of_features))




num_test = 0.20
seed = 10
(X_all, y_all) = (train_df.drop("TARGET", axis = 1, inplace = False), train_df["TARGET"])
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, random_state=seed)

model_0 = xgb.XGBClassifier(learning_rate = 0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1, seed = seed).fit(X_test, y_test, verbose = True, eval_metric="auc")
xgb.plot_importance(model_0)




feature_importances = pd.Series(data=model_0.feature_importances_, index=X_test.columns.tolist())
print(feature_importances.sort_values(ascending = False))
important_features = feature_importances.sort_values(ascending = False)[:5].index
print(important_features)




#train_df.boxplot(column = important_features[0])
#plt.figure(figsize=(40, 40))
train_df["var38"].hist(bins=1000)
#print(train_df)




train_df.loc[:, 'var38'].map(np.log).hist(bins=100)




train_df["var38"].value_counts()




train_df.loc[~np.isclose(train_df.var38, 117310.979016), "var38"].mean()




train_df.loc[~np.isclose(train_df.var38, 117310.979016), 'var38'].map(np.log).hist(bins=100)




train_df["var38_new"] = train_df["var38"].where(~np.isclose(train_df.var38, 117310.979016))
train_df["var38_log"] = train_df.var38_new.map(np.log)
#np.isclose(train_df.var38, 117310.979016)
#train_df["logvar38"] = train_df.loc[~train_df["var38_new"], "var38"].map(np.log)
#train_df.loc[train_df["var38_new"], "logvar38"] = 0




train_df.var38_new.map(np.log).hist(bins=100)




sns.FacetGrid(train_df, hue="TARGET", size=6)    .map(sns.kdeplot, "var38_log")    .add_legend()
plt.title('Density plot of var38_log');




train_df["var15"].hist(bins = 100)




train_df.var15.map(np.log).hist(bins= 100)




print(train_df["var15"].value_counts())





sns.FacetGrid(train_df, hue="TARGET", size=6)    .map(sns.kdeplot, "var15")    .add_legend()
plt.title('Density plot of var15');




train_df.saldo_medio_var5_hace2.hist(bins = 100)




train_df.saldo_medio_var5_hace2.hist(bins = 50)
train_df.saldo_medio_var5_hace2.describe()




print(train_df.saldo_medio_var5_hace2.value_counts())




print("Number of negative entries of saldo_medio_var5_hace2: ", (train_df["saldo_medio_var5_hace2"] < 0).sum())
train_df['log_saldo_medio_var5_hace2'] = train_df.saldo_medio_var5_hace2.map(np.log1p)




sns.FacetGrid(train_df, hue="TARGET", size=10).map(sns.kdeplot, "log_saldo_medio_var5_hace2").add_legend()
plt.title('Density plot of log_saldo_medio_var5_hace2');




train_df.saldo_medio_var5_hace3.hist(bins = 50)
train_df.saldo_medio_var5_hace3.describe()




train_df.saldo_medio_var5_hace3.value_counts()




train_df['log_saldo_medio_var5_hace3'] = train_df.saldo_medio_var5_hace3.map(np.log1p)
train_df['log_saldo_medio_var5_hace3'].dropna().hist(bins = 100)




sns.FacetGrid(train_df, hue="TARGET", size=10).map(sns.kdeplot, "log_saldo_medio_var5_hace3").add_legend()
plt.title("Density plot of log_saldo_medio_var5_hace3");




train_df.saldo_var30.hist(bins = 100)
train_df.saldo_var30.describe()




print(train_df.saldo_var30.value_counts())




train_df['log_saldo_var30'] = train_df.saldo_var30.map(np.log1p)
train_df['log_saldo_var30'].dropna().hist(bins = 100)




sns.FacetGrid(train_df, hue="TARGET", size=10).map(sns.kdeplot, "log_saldo_var30").add_legend()
plt.title('Density plot of saldo_var30')




#creating train set with new and transformed features
#print(train_df)
X_all_new = train_df.drop(["TARGET", "var38", "log_saldo_medio_var5_hace2", "log_saldo_medio_var5_hace3",                           "log_saldo_var30", "var38_log"], axis = 1, inplace = False)
y_all = train_df["TARGET"]
#splitting for training and model validation
seed = 12
X_train_old, X_test_old, y_train, y_test = train_test_split(X_all, y_all, random_state=seed)
eval_set_old = [(X_train_old, y_train), (X_test_old, y_test)]
X_train_new, X_test_new, y_train, y_test = train_test_split(X_all_new, y_all, random_state=seed)
eval_set_new = [(X_train_new, y_train), (X_test_new, y_test)]

#constructing models
model1_old = xgb.XGBClassifier(learning_rate = 0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1, seed = seed).fit(X_test_old, y_test, verbose = True, eval_metric="auc")

model1_new = xgb.XGBClassifier(learning_rate = 0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1, seed = seed).fit(X_test_new, y_test, verbose = True, eval_metric="auc")




model1_old.fit(X_train_old, y_train, eval_metric="auc",               eval_set=eval_set_old, verbose=True, early_stopping_rounds=10)




xgb.plot_importance(model1_old)




model1_new.fit(X_train_new, y_train, eval_metric="auc", eval_set=eval_set_new,               verbose=True, early_stopping_rounds=10)




xgb.plot_importance(model1_new)




feature_importances = pd.Series(data=model1_new.feature_importances_, index=X_test_new.columns.tolist())
feature_importances.sort_values(ascending = False, inplace = True)




seed = 11
result_best = -1
best_n = 0
for features_n in range(3,31):
    features = feature_importances.index[:features_n]
    print("Number of features: ", features_n)
    X_all_reduced = X_all_new.loc[:,features]
    X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(X_all_reduced, y_all, random_state=seed)
    eval_set_reduced = [(X_train_reduced, y_train), (X_test_reduced, y_test)]

#constructing models
    model1_reduced = xgb.XGBClassifier(learning_rate = 0.1, n_estimators=100, max_depth=5,         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,         objective= 'binary:logistic', nthread=4, scale_pos_weight=1,         seed = seed)
    model1_reduced.fit(X_train_reduced, y_train, eval_metric="auc", eval_set=eval_set_reduced,               early_stopping_rounds=10)
    results = model1_reduced.evals_result()
    result_new = results['validation_1']['auc'][-1]
    print("AUC: ", result_new)
    if result_new > result_best:
        result_best = result_new
        best_n = features_n
print("Best AUC achieved with %d features." % best_n)
    




seed = 3
features = feature_importances.index[:best_n + 1]
X_reduced_final = X_all_new.loc[:,features]
model_final = xgb.XGBClassifier(learning_rate = 0.1, n_estimators=100, max_depth=5,         min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,         objective= 'binary:logistic', nthread=4, scale_pos_weight=1,         seed = seed)
model_final.fit(X_reduced_final, y_all)




test_df["var38_new"] = test_df["var38"].where(~np.isclose(test_df.var38, 117310.979016))
X_test = test_df.loc[:,features]




predictions = model_final.predict_proba(X_test)
preds = [round(x[1]) for x in predictions]
print("Predicted precentage of unsatisfied customers: ", sum(preds)/len(preds))




submission = pd.DataFrame(data = predictions[:,1], index = X_test.index, columns = ["TARGET"])
submission.to_csv("prediction_small.csv")




X_all_train = train_df.drop(["TARGET", "var38", "log_saldo_medio_var5_hace2", "log_saldo_medio_var5_hace3",                           "log_saldo_var30", "var38_log"], axis = 1)
y = train_df["TARGET"]
X_all_test = test_df.loc[:,pd.Index(X_all_train.columns)]

model2 = xgb.XGBClassifier(learning_rate = 0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1, seed = seed).fit(X_all_train, y)
predictions = model2.predict_proba(X_all_test)




#print((predictions == 0).sum() + (predictions == 1).sum() - len(predictions))
preds = [round(x[1]) for x in predictions]
print(predictions)
print("Predicted precentage of unsatisfied customers: ", sum(preds)/len(preds))




submission = pd.DataFrame(data = predictions[:,1], index = X_test.index, columns = ["TARGET"])
submission.to_csv("prediction_full.csv")

