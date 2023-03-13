#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb

from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


color = sns.color_palette()

#%matplotlib inline

pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 500)
run1 = False
run2 = False
run3 = False




train_df = pd.read_csv("../input/train.csv")
train_df.shape




train_df.head()




print(train_df.shape[0])




plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.price_doc.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('price', fontsize=12)
plt.show()




plt.figure(figsize=(12,8))
dfsnsplt = pd.DataFrame(train_df.price_doc.values.astype(int))
print(type(dfsnsplt))
sns.distplot(dfsnsplt, bins=60, kde=True)
#sns.distplot(train_df.price_doc.values)
plt.xlabel('price', fontsize=12)
plt.show()




plt.figure(figsize=(12,8))
sns.distplot(np.log(train_df.price_doc.values), bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()




train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()




plt.figure(figsize=(12,8))
sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Year Month', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()




train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()




missing_df_temp = train_df.isnull()




missing_df_temp.head()




missing_df_temp.sum(axis=0).reset_index().head()




missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='y')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()




df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
id_test = list(df_test['id'])




for f in df_test.columns:
    if df_test[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(df_test[f].values)) 
        df_test[f] = lbl.transform(list(df_test[f].values))




#clean data
bad_index = train_df[train_df.life_sq > train_df.full_sq].index
train_df.ix[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
df_test.ix[equal_index, "life_sq"] = df_test.ix[equal_index, "full_sq"]
bad_index = df_test[df_test.life_sq > df_test.full_sq].index
df_test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train_df[train_df.life_sq < 5].index
train_df.ix[bad_index, "life_sq"] = np.NaN
bad_index = df_test[df_test.life_sq < 5].index
df_test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train_df[train_df.full_sq < 5].index
train_df.ix[bad_index, "full_sq"] = np.NaN
bad_index = df_test[df_test.full_sq < 5].index
df_test.ix[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train_df.ix[kitch_is_build_year, "build_year"] = train_df.ix[kitch_is_build_year, "kitch_sq"]
bad_index = train_df[train_df.kitch_sq >= train_df.life_sq].index
train_df.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = df_test[df_test.kitch_sq >= df_test.life_sq].index
df_test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train_df[(train_df.kitch_sq == 0).values + (train_df.kitch_sq == 1).values].index
train_df.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = df_test[(df_test.kitch_sq == 0).values + (df_test.kitch_sq == 1).values].index
df_test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train_df[(train_df.full_sq > 210) & (train_df.life_sq / train_df.full_sq < 0.3)].index
train_df.ix[bad_index, "full_sq"] = np.NaN
bad_index = df_test[(df_test.full_sq > 150) & (df_test.life_sq / df_test.full_sq < 0.3)].index
df_test.ix[bad_index, "full_sq"] = np.NaN
bad_index = train_df[train_df.life_sq > 300].index
train_df.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = df_test[df_test.life_sq > 200].index
df_test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
train_df.product_type.value_counts(normalize= True)
df_test.product_type.value_counts(normalize= True)
bad_index = train_df[train_df.build_year < 1500].index
train_df.ix[bad_index, "build_year"] = np.NaN
bad_index = df_test[df_test.build_year < 1500].index
df_test.ix[bad_index, "build_year"] = np.NaN
bad_index = train_df[train_df.num_room == 0].index 
train_df.ix[bad_index, "num_room"] = np.NaN
bad_index = df_test[df_test.num_room == 0].index 
df_test.ix[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train_df.ix[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
df_test.ix[bad_index, "num_room"] = np.NaN
bad_index = train_df[(train_df.floor == 0).values * (train_df.max_floor == 0).values].index
train_df.ix[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train_df[train_df.floor == 0].index
train_df.ix[bad_index, "floor"] = np.NaN
bad_index = train_df[train_df.max_floor == 0].index
train_df.ix[bad_index, "max_floor"] = np.NaN
bad_index = df_test[df_test.max_floor == 0].index
df_test.ix[bad_index, "max_floor"] = np.NaN
bad_index = train_df[train_df.floor > train_df.max_floor].index
train_df.ix[bad_index, "max_floor"] = np.NaN
bad_index = df_test[df_test.floor > df_test.max_floor].index
df_test.ix[bad_index, "max_floor"] = np.NaN
train_df.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train_df.ix[bad_index, "floor"] = np.NaN
train_df.material.value_counts()
df_test.material.value_counts()
train_df.state.value_counts()
bad_index = train_df[train_df.state == 33].index
train_df.ix[bad_index, "state"] = np.NaN
df_test.state.value_counts()




ulimit = np.percentile(train_df.price_doc.values, 99.5)
llimit = np.percentile(train_df.price_doc.values, 0.5)
train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit

col = "full_sq"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit





# Add month-year
"""
month_year = (train_df.timestamp.dt.month.astype(int) + (train_df.timestamp.dt.year * 100).astype(int))
month_year_cnt_map = month_year.value_counts().to_dict()
train_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

month_year = (df_test.timestamp.dt.month.astype(int) + (df_test.timestamp.dt.year * 100).astype(int))
month_year_cnt_map = month_year.value_counts().to_dict()
df_test['month_year_cnt'] = month_year.map(month_year_cnt_map)


# Add week-year count
week_year = (train_df.timestamp.dt.weekofyear.astype(int) + (train_df.timestamp.dt.year * 100).astype(int))
week_year_cnt_map = week_year.value_counts().to_dict()
train_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

week_year = (df_test.timestamp.dt.weekofyear.astype(int) + (df_test.timestamp.dt.year * 100).astype(int))
week_year_cnt_map = week_year.value_counts().to_dict()
df_test['week_year_cnt'] = week_year.map(week_year_cnt_map)


# Add month and day-of-week
train_df['month'] = train_df.timestamp.dt.month
train_df['dow'] = train_df.timestamp.dt.dayofweek

df_test['month'] = df_test.timestamp.dt.month
df_test['dow'] = df_test.timestamp.dt.dayofweek


# Other feature engineering
train_df['rel_floor'] = train_df['floor'] / train_df['max_floor'].astype(float)
train_df['rel_kitch_sq'] = train_df['kitch_sq'] / train_df['full_sq'].astype(float)

df_test['rel_floor'] = df_test['floor'] / df_test['max_floor'].astype(float)
df_test['rel_kitch_sq'] = df_test['kitch_sq'] / df_test['full_sq'].astype(float)

train_df.apartment_name=train_df.sub_area + train_df['metro_km_avto'].astype(str)
df_test.apartment_name=df_test.sub_area + train_df['metro_km_avto'].astype(str)

train_df['room_size'] = train_df['life_sq'] / train_df['num_room'].astype(float)
df_test['room_size'] = df_test['life_sq'] / df_test['num_room'].astype(float)
"""




for f in train_df.columns:
    if train_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values)) 
        train_df[f] = lbl.transform(list(train_df[f].values))
        
train_y = train_df.price_doc.values
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)




train_df.columns.values




print(train_X.shape)
print(train_y.shape)




#model gradient boost

#clfdata_X = pd.DataFrame(np.nan_to_num(train_df.drop(['id','timestamp','price_doc'],axis=1)))
#clfdata_y = pd.DataFrame(np.nan_to_num(train_df['price_doc']))

#train_X, X_val, train_y, y_val = train_test_split(clfdata_X, clfdata_y, test_size=0.30,random_state=21)

train_X1 = np.nan_to_num(train_X[:25000]).astype(int)
val_X = np.nan_to_num(train_X[25000:]).astype(int)
train_y1 = np.nan_to_num(train_y[:25000]).astype(int)
val_y = np.nan_to_num(train_y[25000:]).astype(int)




GBclf= GradientBoostingClassifier(max_depth=4,min_samples_leaf=2)




clfX_train.head(1)




train_y1[0]




#train_X, X_val, train_y, y_val

GBclf.fit(train_X1,train_y1)
GBclf.score(val_X,val_y)




#predict = GBmodel.predict(test_df.drop(["id", "timestamp"],axis=1))
predict = GBclf.predict(test_df.drop(['id','timestamp'],axis=1))
output = pd.DataFrame({'id': id_test, 'price_doc': np.expm1(predict)})
#output['price_doc'] = lab




output.to_csv('Sberbank_GBclf.csv', index=False)




if run1 == True:

    xgb_params = {
        'eta': 0.05,
        'max_depth': 6,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

    # plot the important features #
    fig, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()




if run1 == True:
    print(id_test[:10])




if run1 == True:
    df_test.drop(["id", "timestamp"], axis=1, inplace=True)




if run1 == True:
    dtest = xgb.DMatrix(df_test, feature_names=train_X.columns)

    y_pred = model.predict(dtest)

    y_pred = np.round(y_pred * 1.008)
    df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

    df_sub.to_csv('Sberbank_0.csv', index=False)




df_sub.head()




if run2 == True:
    ulimit = np.percentile(train_df.price_doc.values, 99.5)
    llimit = np.percentile(train_df.price_doc.values, 0.5)
    train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit
    train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit

    col = "full_sq"
    ulimit = np.percentile(train_df[col].values, 99.5)
    llimit = np.percentile(train_df[col].values, 0.5)
    train_df[col].ix[train_df[col]>ulimit] = ulimit
    train_df[col].ix[train_df[col]<llimit] = llimit

    plt.figure(figsize=(12,12))
    sns.jointplot(x=np.log1p(train_df.full_sq.values), y=np.log1p(train_df.price_doc.values), size=10)
    plt.ylabel('Log of Price', fontsize=12)
    plt.xlabel('Log of Total area in square metre', fontsize=12)
    plt.show()




if run2 == True:
    col = "life_sq"
    train_df[col].fillna(0, inplace=True)
    ulimit = np.percentile(train_df[col].values, 95)
    llimit = np.percentile(train_df[col].values, 5)
    train_df[col].ix[train_df[col]>ulimit] = ulimit
    train_df[col].ix[train_df[col]<llimit] = llimit

    plt.figure(figsize=(12,12))
    sns.jointplot(x=np.log1p(train_df.life_sq.values), y=np.log1p(train_df.price_doc.values), 
              kind='kde', size=10)
    plt.ylabel('Log of Price', fontsize=12)
    plt.xlabel('Log of living area in square metre', fontsize=12)
    plt.show()




if run2 == True:
    plt.figure(figsize=(12,8))
    sns.countplot(x="floor", data=train_df)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('floor number', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()




if run2 == True:
    grouped_df = train_df.groupby('floor')['price_doc'].aggregate(np.median).reset_index()
    plt.figure(figsize=(12,8))
    sns.pointplot(grouped_df.floor.values, grouped_df.price_doc.values, alpha=0.8, color=color[2])
    plt.ylabel('Median Price', fontsize=12)
    plt.xlabel('Floor number', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()




if run2 == True:
    plt.figure(figsize=(12,8))
    sns.countplot(x="max_floor", data=train_df)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Max floor number', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()




if run2 == True:
    plt.figure(figsize=(12,8))
    sns.boxplot(x="max_floor", y="price_doc", data=train_df)
    plt.ylabel('Median Price', fontsize=12)
    plt.xlabel('Max Floor number', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()




if run3 == True:
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import matplotlib.pyplot as plt




if run3 == True:
    df_train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
    df_test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
    df_macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])




if run3 == True:
    df_train.head()




if run3 == True:
    # cleanup
    print(df_train.shape)
    df_train.loc[df_train.full_sq == 0, 'full_sq'] = 30
    df_train = df_train[df_train.price_doc/df_train.full_sq <= 600000]
    df_train = df_train[df_train.price_doc/df_train.full_sq >= 10000]
    print(df_train.shape)




# if run3 == True:
    #print(df_train.loc[df_train.full_sq == 30, 'full_sq'])




if run3 == True:
    print(df_test.shape+df_train.shape)
    #print(df_train.shape)




# if run3 == True:
    #print(df_test['id'][:10])
    #df_test.head()
    #df_train.head()




# if run3 == True:
    #print(df_train.loc[df_train.full_sq == 30, 'price_doc'],  df_train.loc[df_train.full_sq == 30, 'full_sq'])




if run3 == True:
    y_train = df_train['price_doc'].values
    id_test = df_test['id']

    df_train.drop(['id', 'price_doc'], axis=1, inplace=True)
    df_test.drop(['id'], axis=1, inplace=True)

    # Build df_all = (df_train+df_test).join(df_macro)
    num_train = len(df_train)
    df_all = pd.concat([df_train, df_test])
    df_all = df_all.join(df_macro, on='timestamp', rsuffix='_macro')
    print(df_all.shape)




# if run3 == True:
    #df_all.describe




if run3 == True:
    # Add month-year
    #year= df_all.timestamp.dt.year
    #year_cnt_map = year.value_counts().to_dict()
    #df_all['year_cnt'] = year.map(year_cnt_map)
    #df_all['Age_building']=2018-df_all['build_year']

    # Other feature engineering
    df_all['rel_floor'] = df_all['floor'] / df_all['max_floor'].astype(float)
    df_all['rel_kitch_sq'] = df_all['kitch_sq'] / df_all['full_sq'].astype(float)

    # Remove timestamp column (may overfit the model in train)
    df_all.drop(['timestamp', 'timestamp_macro'], axis=1, inplace=True)




if run3 == True:
    factorize = lambda t: pd.factorize(t[1])[0]




if run3 == True:
    df_obj = df_all.select_dtypes(include=['object'])




if run3 == True:
    df_obj.shape




if run3 == True:
    z = np.array(list(map(factorize, df_obj.iteritems()))).T




if run3 == True:
    q = np.array([x for x in df_obj.iteritems()]).T




if run3 == True:
    print(q[:10])




# if run3 == True:
    #df_obj = df_all.select_dtypes(include=['object'])

    #X_all = np.c_[
    #    df_all.select_dtypes(exclude=['object']).values,
    #    np.array(list(map(factorize, df_obj.iteritems()))).T
    #]
    #print(X_all.shape)




# if run3 == True:
    #X_train = X_all[:num_train]
    #X_test = X_all[num_train:]




if run3 == True:
    # Deal with categorical values
    df_numeric = df_all.select_dtypes(exclude=['object'])
    df_obj = df_all.select_dtypes(include=['object']).copy()

    for c in df_obj:
        df_obj[c] = pd.factorize(df_obj[c])[0]

    df_values = pd.concat([df_numeric, df_obj], axis=1)


    # Convert to numpy values
    X_all = df_values.values
    print(X_all.shape)

    X_train = X_all[:num_train]
    X_test = X_all[num_train:]

    df_columns = df_values.columns
    X_train=np.nan_to_num(X_train)
    X_test=np.nan_to_num(X_test)




if run3 == True:
    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(X_train, y_train, feature_names=df_columns)
    dtest = xgb.DMatrix(X_test, feature_names=df_columns)




if run3 == True:
    # Uncomment to tune XGB `num_boost_rounds`

    #cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    #    verbose_eval=True, show_stdv=False)
    #cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
    #num_boost_rounds = len(cv_result)

    num_boost_round = 489

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_round)




if run3 == True:
    fig, ax = plt.subplots(figsize=(12,18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()




if run3 == True:
    #fig, ax = plt.subplots(1, 1, figsize=(8, 16))
    #xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

    y_pred = model.predict(dtest)
    y_pred = np.round(y_pred * 1.008)
    df_sub = pd.DataFrame({'id': id_test, 'price_doc': y_pred})

    df_sub.to_csv('Sberbank_1.csv', index=False)




if run3 == True:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn import model_selection, preprocessing
    import xgboost as xgb
    import datetime




if run3 == True:
    #load files
    train = pd.read_csv('../input/train.csv', parse_dates=['timestamp'])
    test = pd.read_csv('../input/test.csv', parse_dates=['timestamp'])
    macro = pd.read_csv('../input/macro.csv', parse_dates=['timestamp'])
    id_test = test.id

    #multiplier = 0.969




if run3 == True:
    #clean data
    bad_index = train[train.life_sq > train.full_sq].index
    train.ix[bad_index, "life_sq"] = np.NaN
    equal_index = [601,1896,2791]
    test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
    bad_index = test[test.life_sq > test.full_sq].index
    test.ix[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.life_sq < 5].index
    train.ix[bad_index, "life_sq"] = np.NaN
    bad_index = test[test.life_sq < 5].index
    test.ix[bad_index, "life_sq"] = np.NaN
    bad_index = train[train.full_sq < 5].index
    train.ix[bad_index, "full_sq"] = np.NaN
    bad_index = test[test.full_sq < 5].index
    test.ix[bad_index, "full_sq"] = np.NaN
    kitch_is_build_year = [13117]
    train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
    bad_index = train[train.kitch_sq >= train.life_sq].index
    train.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[test.kitch_sq >= test.life_sq].index
    test.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
    train.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
    test.ix[bad_index, "kitch_sq"] = np.NaN
    bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
    train.ix[bad_index, "full_sq"] = np.NaN
    bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
    test.ix[bad_index, "full_sq"] = np.NaN
    bad_index = train[train.life_sq > 300].index
    train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
    bad_index = test[test.life_sq > 200].index
    test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
    train.product_type.value_counts(normalize= True)
    test.product_type.value_counts(normalize= True)
    bad_index = train[train.build_year < 1500].index
    train.ix[bad_index, "build_year"] = np.NaN
    bad_index = test[test.build_year < 1500].index
    test.ix[bad_index, "build_year"] = np.NaN
    bad_index = train[train.num_room == 0].index 
    train.ix[bad_index, "num_room"] = np.NaN
    bad_index = test[test.num_room == 0].index 
    test.ix[bad_index, "num_room"] = np.NaN
    bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
    train.ix[bad_index, "num_room"] = np.NaN
    bad_index = [3174, 7313]
    test.ix[bad_index, "num_room"] = np.NaN
    bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
    train.ix[bad_index, ["max_floor", "floor"]] = np.NaN
    bad_index = train[train.floor == 0].index
    train.ix[bad_index, "floor"] = np.NaN
    bad_index = train[train.max_floor == 0].index
    train.ix[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.max_floor == 0].index
    test.ix[bad_index, "max_floor"] = np.NaN
    bad_index = train[train.floor > train.max_floor].index
    train.ix[bad_index, "max_floor"] = np.NaN
    bad_index = test[test.floor > test.max_floor].index
    test.ix[bad_index, "max_floor"] = np.NaN
    train.floor.describe(percentiles= [0.9999])
    bad_index = [23584]
    train.ix[bad_index, "floor"] = np.NaN
    train.material.value_counts()
    test.material.value_counts()
    train.state.value_counts()
    bad_index = train[train.state == 33].index
    train.ix[bad_index, "state"] = np.NaN
    test.state.value_counts()




if run3 == True:
    # brings error down a lot by removing extreme price per sqm
    train.loc[train.full_sq == 0, 'full_sq'] = 50
    train = train[train.price_doc/train.full_sq <= 600000]
    train = train[train.price_doc/train.full_sq >= 10000]




if run3 == True:
    # Add month-year
    month_year = (train.timestamp.dt.month + train.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    train['month_year_cnt'] = month_year.map(month_year_cnt_map)

    month_year = (test.timestamp.dt.month + test.timestamp.dt.year * 100)
    month_year_cnt_map = month_year.value_counts().to_dict()
    test['month_year_cnt'] = month_year.map(month_year_cnt_map)




if run3 == True:
    # Add week-year count
    week_year = (train.timestamp.dt.weekofyear + train.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    train['week_year_cnt'] = week_year.map(week_year_cnt_map)

    week_year = (test.timestamp.dt.weekofyear + test.timestamp.dt.year * 100)
    week_year_cnt_map = week_year.value_counts().to_dict()
    test['week_year_cnt'] = week_year.map(week_year_cnt_map)




if run3 == True:
    # Add month and day-of-week
    train['month'] = train.timestamp.dt.month
    train['dow'] = train.timestamp.dt.dayofweek

    test['month'] = test.timestamp.dt.month
    test['dow'] = test.timestamp.dt.dayofweek




if run3 == True:
    # Other feature engineering
    train['rel_floor'] = train['floor'] / train['max_floor'].astype(float)
    train['rel_kitch_sq'] = train['kitch_sq'] / train['full_sq'].astype(float)

    test['rel_floor'] = test['floor'] / test['max_floor'].astype(float)
    test['rel_kitch_sq'] = test['kitch_sq'] / test['full_sq'].astype(float)

    train.apartment_name=train.sub_area + train['metro_km_avto'].astype(str)
    test.apartment_name=test.sub_area + train['metro_km_avto'].astype(str)

    train['room_size'] = train['life_sq'] / train['num_room'].astype(float)
    test['room_size'] = test['life_sq'] / test['num_room'].astype(float)




if run3 == True:
    y_train = train["price_doc"]
    x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
    x_test = test.drop(["id", "timestamp"], axis=1)

    for c in x_train.columns:
        if x_train[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_train[c].values)) 
            x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)
        
    for c in x_test.columns:
        if x_test[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(x_test[c].values)) 
            x_test[c] = lbl.transform(list(x_test[c].values))
            #x_test.drop(c,axis=1,inplace=True)  




if run3 == True:
    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    dtrain = xgb.DMatrix(x_train, y_train)
    dtest = xgb.DMatrix(x_test)




if run3 == True:
    #cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    #    verbose_eval=50, show_stdv=False)
    #cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

    #num_boost_rounds = len(cv_output)
    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= 350)

    #fig, ax = plt.subplots(1, 1, figsize=(8, 13))
    #xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)

    y_predict = model.predict(dtest)
    y_predict = np.round(y_predict * 0.99)
    output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
    output.head()




if run3 == True:
    output.to_csv('Sberbank_2.csv', index=False)

