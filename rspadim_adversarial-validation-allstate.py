import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

from sklearn.metrics import roc_auc_score



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# We start by loading the training / test data and combining them with minimal preprocessing necessary

xtrain = pd.read_csv('../input/train.csv')

xtrain.drop(['id', 'loss'], axis = 1, inplace = True)

xtest = pd.read_csv('../input/test.csv')

xtest.drop(['id'], axis = 1, inplace = True)



# add identifier and combine

xtrain['istrain'] = 1

xtest['istrain'] = 0

xdat = pd.concat([xtrain, xtest], axis = 0)

# convert non-numerical columns to integers

df_numeric = xdat.select_dtypes(exclude=['object'])

df_obj = xdat.select_dtypes(include=['object']).copy()

    

for c in df_obj:

    df_obj[c] = pd.factorize(df_obj[c])[0]

    

xdat = pd.concat([df_numeric, df_obj], axis=1)

y = xdat['istrain']; xdat.drop('istrain', axis = 1, inplace = True)
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 44) # why stratified k fold?

xgb_params = { # is this parameters ok?

        'learning_rate': 0.05, 'max_depth': 4,'subsample': 0.9,

        'colsample_bytree': 0.9,'objective': 'binary:logistic',

        'silent': 1, 'n_estimators':100, 'gamma':1,

        'min_child_weight':4, 'n_jobs':-1

        }   

clf = xgb.XGBClassifier(**xgb_params, seed = 10)     
for train_index, test_index in skf.split(xdat, y):

        x0, x1 = xdat.iloc[train_index], xdat.iloc[test_index]

        y0, y1 = y.iloc[train_index], y.iloc[test_index]        

        print(x0.shape)

        clf.fit(x0, y0, eval_set=[(x1, y1)],

               eval_metric='logloss', verbose=False,early_stopping_rounds=10) # it takes ~ 80 rounds to fit

                

        prval = clf.predict_proba(x1)[:,1]

        print(roc_auc_score(y1,prval))

        

#final dataset:

clf.fit(xdat, y, eval_set=[(x1, y1)],

eval_metric='logloss', verbose=False,early_stopping_rounds=10) # it takes ~ 80 rounds to fit



prval = clf.predict_proba(xdat)[:,1]

print(roc_auc_score(y,prval))
from sklearn.neighbors import KNeighborsClassifier

knn_params={

    'n_neighbors':5, # first try value

    'weights':'distance',

    'metric':'manhattan' #i like this name =)

    

    #distances to test: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

    #float:            

    #   euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis

    #integers: 

    #   hamming, canberra, braycurtis

}



clf = KNeighborsClassifier(**knn_params)      #good bye xgboost
for train_index, test_index in skf.split(xdat, y):

        x0, x1 = xdat.iloc[train_index], xdat.iloc[test_index]

        y0, y1 = y.iloc[train_index], y.iloc[test_index]        

        print(x0.shape)

        clf.fit(x0, y0) # very easy parameters :)

                

        prval = clf.predict_proba(x1)[:,1]

        print(roc_auc_score(y1,prval))

        

#final dataset:

clf.fit(xdat, y)



prval = clf.predict_proba(xdat)[:,1]

print(roc_auc_score(y,prval))