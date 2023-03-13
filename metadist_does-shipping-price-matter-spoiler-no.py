#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd 




train = pd.read_csv('../input/train.tsv',delimiter='\t', index_col='train_id')
test = pd.read_csv('../input/test.tsv',delimiter='\t', index_col='test_id')




#fill in NaNs
train['brand_name'].fillna('NONAME', inplace=True)
train['category_name'].fillna('NOCAT', inplace=True)




#split train into train and validation set, 
#we will fit the model on log price from the begining
from sklearn.model_selection import train_test_split

X_train_part, X_valid, y_train_part, y_valid =     train_test_split(train, np.log1p(train['price'].values), random_state=17)




from catboost import CatBoostRegressor
#Set some pre-tuned parameters. #Want better score - increase interations.
I=200; lr = 0.5
cb_params={'has_time':False, 'eval_metric':'RMSE', 'logging_level':'Silent', 'train_dir':'/tmp'}
cb_fit_columns=['item_condition_id', 'category_name', 'brand_name','shipping']
cb_fit_params={'cat_features':[0,1,2,3]} #all train features are categorical
cb=CatBoostRegressor(**cb_params,iterations=I, learning_rate=lr)
cb.fit(X_train_part[cb_fit_columns], y_train_part, **cb_fit_params)




from sklearn.metrics import mean_squared_error
import math
pred = cb.predict(X_valid[cb_fit_columns])
print(math.sqrt(mean_squared_error(y_valid, pred )))




cb_fit_columns=['item_condition_id', 'category_name', 'brand_name']
cb_fit_params={'cat_features':[0,1,2]} #all train features are categorical
cb=CatBoostRegressor(**cb_params,iterations=I, learning_rate=lr)
cb.fit(X_train_part[cb_fit_columns], y_train_part, **cb_fit_params)




pred_no_ship = cb.predict(X_valid[cb_fit_columns])
print(math.sqrt(mean_squared_error(y_valid, pred_no_ship )))




X_valid_pred = X_valid.copy()
X_valid_pred['pred'] = np.expm1(pred_no_ship) #restore to "human" scale

#to keep track of price changes
ship_surcharges = []
score_vs_ship_surcharge = []

for ship_surcharge in np.arange(0.0, 4.0, 0.1):
    X_valid_pred["pred_shipped"] = X_valid_pred['pred']
    #We only amend thices with shipping by the bayer
    X_valid_pred.loc[X_valid_pred['shipping']==0,"pred_shipped"] += ship_surcharge
    
    pred_mod_score = math.sqrt(mean_squared_error(y_valid, #np.log1p(y_valid),
                                                  np.log1p(X_valid_pred['pred_shipped'].clip(0))))
    #
    ship_surcharges.append(ship_surcharge)
    score_vs_ship_surcharge.append(pred_mod_score)
    
best_score = min(score_vs_ship_surcharge)
best_ship = ship_surcharges[np.argmin(score_vs_ship_surcharge)]
print(best_ship, best_score)




from matplotlib import pyplot as plt
plt.plot(ship_surcharges, score_vs_ship_surcharge)




X_valid_pred = X_valid.copy()
X_valid_pred['pred'] = np.expm1(pred) #restore to "human" scale

#to keep track of price changes
ship_surcharges = []
score_vs_ship_surcharge = []

for ship_surcharge in np.arange(-3.0, 4.0, 0.1):
    X_valid_pred["pred_shipped"] = X_valid_pred['pred']
    #We only amend thices with shipping by the bayer
    X_valid_pred.loc[X_valid_pred['shipping']==0,"pred_shipped"] += ship_surcharge
    
    pred_mod_score = math.sqrt(mean_squared_error(y_valid, #np.log1p(y_valid),
                                                  np.log1p(X_valid_pred['pred_shipped'].clip(0))))
    #
    ship_surcharges.append(ship_surcharge)
    score_vs_ship_surcharge.append(pred_mod_score)
    
best_score = min(score_vs_ship_surcharge)
best_ship = ship_surcharges[np.argmin(score_vs_ship_surcharge)]
print(best_ship, best_score)




plt.plot(ship_surcharges, score_vs_ship_surcharge)




#optimize this function of score vs. shipping surcharge (s)
def f(s, X, y):
    X["pred_mod"] = X['pred'].copy()
    X.loc[X['shipping']==0,"pred_mod"] += s
    score = math.sqrt(mean_squared_error(y, 
                                         np.log1p(X['pred_mod'].clip(0))))
    return score




#I take the first optimizer from the list, but others may yield better results
from scipy import optimize
min_s = optimize.minimize(f, 0, args=(X_valid_pred, y_valid), method='Nelder-Mead')




#We got the same values - good!
min_s.x[0], min_s.fun




#let's chose 10 most frequent categories,
#otherwise our optimization process takes too long.
all_cats = X_valid.groupby('category_name')['name'].count().to_frame()    .sort_values(by='name', ascending=False).index




X_valid_pred = X_valid.copy()
X_valid_pred['pred'] = np.expm1(pred_no_ship) #restore to "human" scale
X_valid_pred['pred_opt'] = np.expm1(pred_no_ship) 

MIN0=1.9 #initial value for the optimizer. Recall, this is our best average across all categories
NCAT=10 # number of categories to optimize. this is enough to get an idea.
METHOD='Nelder-Mead'

scores_cat = []

for cat in all_cats[:NCAT]:
    mask = X_valid_pred['category_name']==cat
    res = optimize.minimize(f, MIN0, args=(X_valid_pred[mask], y_valid[mask]), method=METHOD)
    print("{0:>2.2f} {1:1.3f} {2:s}".format (res.x[0],res.fun, cat))
    #update predicted values                        
    X_valid_pred.loc[mask,'pred_opt'] = X_valid_pred[mask]['pred'] + res.x[0]
    #calculate the total score with price adjustment with current category included.
    scores_cat.append( math.sqrt(mean_squared_error(y_valid, 
                                    np.log1p(X_valid_pred['pred_opt'].clip(0)))) )




X_valid_pred[X_valid_pred['category_name'].isin(all_cats[:NCAT])][['category_name','price','pred','pred_opt']].head(10)




#score for this optimization
score = math.sqrt(mean_squared_error(y_valid, 
                                    np.log1p(X_valid_pred['pred_opt'].clip(0))))
score




plt.plot(scores_cat)
    




#catboost on full data
math.sqrt(mean_squared_error(y_valid, pred))




#catboost with no shipping column in data
math.sqrt(mean_squared_error(y_valid, pred_no_ship))




# permute the shipping column
train['shipping'] = np.random.permutation(train['shipping'])

X_train_part, X_valid, y_train_part, y_valid =     train_test_split(train, np.log1p(train['price'].values), random_state=17)




#Set some pre-tuned parameters. #Want better score - increase interations.
I=200; lr = 0.5
cb_params={'has_time':False, 'eval_metric':'RMSE', 'logging_level':'Silent', 'train_dir':'/tmp'}
cb_fit_columns=['item_condition_id', 'category_name', 'brand_name','shipping']
cb_fit_params={'cat_features':[0,1,2,3]} #all train features are categorical
cb=CatBoostRegressor(**cb_params,iterations=I, learning_rate=lr)
cb.fit(X_train_part[cb_fit_columns], y_train_part, **cb_fit_params)




from sklearn.metrics import mean_squared_error
import math
pred_ship_permute = cb.predict(X_valid[cb_fit_columns])
print(math.sqrt(mean_squared_error(y_valid, pred_ship_permute)))




#lets retrain on the full dataset , predict the test, and submit
cb.fit(train[cb_fit_columns], np.log1p(train['price'].values), **cb_fit_params)




#fill in NaNs
test['brand_name'].fillna('NONAME', inplace=True)
test['category_name'].fillna('NOCAT', inplace=True)

pred_test=cb.predict(test[cb_fit_columns])




test['price'] = np.expm1(pred_test)
test.loc[test['shipping']==0,"price"] += MIN0




#submission
pd.DataFrame({'price':test['price']}, index=test.index)  .to_csv("submission.csv", index_label="test_id")

