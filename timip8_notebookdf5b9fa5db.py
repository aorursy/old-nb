import numpy as np

import pandas as pd

import lightgbm as lgb

import gc
train = pd.read_csv('../input/train_2016.csv')

prop = pd.read_csv('../input/properties_2016.csv')
for c, dtype in zip(prop.columns, prop.dtypes):	

    if dtype == np.float64:		

        prop[c] = prop[c].astype(np.float32)



df_train = train.merge(prop, how='left', on='parcelid')



x_train = df_train.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode'], axis=1)

y_train = df_train['logerror'].values
df_train.head()
print(x_train.shape, y_train.shape)
train_columns = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:

    x_train[c] = (x_train[c] == True)
del df_train; gc.collect()

split = 90000

x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

x_train = x_train.values.astype(np.float32, copy=False)

x_valid = x_valid.values.astype(np.float32, copy=False)



d_train = lgb.Dataset(x_train, label=y_train)

d_valid = lgb.Dataset(x_valid, label=y_valid)
params = {}

params['learning_rate'] = 0.002

params['boosting_type'] = 'gbdt'

params['objective'] = 'regression'

params['metric'] = 'mae'

params['sub_feature'] = 0.5

params['num_leaves'] = 60

params['min_data'] = 500

params['min_hessian'] = 1



watchlist = [d_valid]

clf = lgb.train(params, d_train, 500, watchlist)



del d_train, d_valid; gc.collect()

del x_train, x_valid; gc.collect()



print("Prepare for the prediction ...")

sample = pd.read_csv('../input/sample_submission.csv')

sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(prop, on='parcelid', how='left')

del sample, prop; gc.collect()

x_test = df_test[train_columns]

del df_test; gc.collect()

for c in x_test.dtypes[x_test.dtypes == object].index.values:

    x_test[c] = (x_test[c] == True)

x_test = x_test.values.astype(np.float32, copy=False)



print("Start prediction ...")

# num_threads > 1 will predict very slow in kernal

clf.reset_parameter({"num_threads":1})

p_test = clf.predict(x_test)



del x_test; gc.collect()



print("Start write result ...")

sub = pd.read_csv('../input/sample_submission.csv')

for c in sub.columns[sub.columns != 'ParcelId']:

    sub[c] = p_test