import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing

import xgboost as xgb

color = sns.color_palette()






pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)
train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

macro_df = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')

test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')

print(train_df.shape, test_df.shape)



# truncate the extreme values in price_doc #

ulimit = np.percentile(train_df.price_doc.values, 99)

llimit = np.percentile(train_df.price_doc.values, 1)

train_df['price_doc'].ix[train_df['price_doc']>ulimit] = ulimit

train_df['price_doc'].ix[train_df['price_doc']<llimit] = llimit
for f in train_df.columns:

    if train_df[f].dtype=='object':

        print(f)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))

        train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))

        test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))
train_df["null_count"] = train_df.isnull().sum(axis=1)

test_df["null_count"] = test_df.isnull().sum(axis=1)



plt.figure(figsize=(14,12))

sns.pointplot(x='null_count', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('null_count', fontsize=12)

plt.xticks(rotation='vertical')

plt.show()
train_df.fillna(-99, inplace=True)

test_df.fillna(-99, inplace=True)
# year and month #

train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month

test_df["yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month



# year and week #

train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear

test_df["yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear



# year #

train_df["year"] = train_df["timestamp"].dt.year

test_df["year"] = test_df["timestamp"].dt.year



# month of year #

train_df["month_of_year"] = train_df["timestamp"].dt.month

test_df["month_of_year"] = test_df["timestamp"].dt.month



# week of year #

train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear

test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear



# day of week #

train_df["day_of_week"] = train_df["timestamp"].dt.weekday

test_df["day_of_week"] = test_df["timestamp"].dt.weekday





plt.figure(figsize=(12,8))

sns.pointplot(x='yearweek', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('yearweek', fontsize=12)

plt.title('Median Price distribution by year and week_num')

plt.xticks(rotation='vertical')

plt.show()



plt.figure(figsize=(12,8))

sns.boxplot(x='month_of_year', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('month_of_year', fontsize=12)

plt.title('Median Price distribution by month_of_year')

plt.xticks(rotation='vertical')

plt.show()



plt.figure(figsize=(12,8))

sns.pointplot(x='week_of_year', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('week of the year', fontsize=12)

plt.title('Median Price distribution by week of year')

plt.xticks(rotation='vertical')

plt.show()



plt.figure(figsize=(12,8))

sns.boxplot(x='day_of_week', y='price_doc', data=train_df)

plt.ylabel('price_doc', fontsize=12)

plt.xlabel('day_of_week', fontsize=12)

plt.title('Median Price distribution by day of week')

plt.xticks(rotation='vertical')

plt.show()
# ratio of living area to full area #

train_df["ratio_life_sq_full_sq"] = train_df["life_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)

test_df["ratio_life_sq_full_sq"] = test_df["life_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)

train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]<0] = 0

train_df["ratio_life_sq_full_sq"].ix[train_df["ratio_life_sq_full_sq"]>1] = 1

test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]<0] = 0

test_df["ratio_life_sq_full_sq"].ix[test_df["ratio_life_sq_full_sq"]>1] = 1



# ratio of kitchen area to living area #

train_df["ratio_kitch_sq_life_sq"] = train_df["kitch_sq"] / np.maximum(train_df["life_sq"].astype("float"),1)

test_df["ratio_kitch_sq_life_sq"] = test_df["kitch_sq"] / np.maximum(test_df["life_sq"].astype("float"),1)

train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]<0] = 0

train_df["ratio_kitch_sq_life_sq"].ix[train_df["ratio_kitch_sq_life_sq"]>1] = 1

test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]<0] = 0

test_df["ratio_kitch_sq_life_sq"].ix[test_df["ratio_kitch_sq_life_sq"]>1] = 1



# ratio of kitchen area to full area #

train_df["ratio_kitch_sq_full_sq"] = train_df["kitch_sq"] / np.maximum(train_df["full_sq"].astype("float"),1)

test_df["ratio_kitch_sq_full_sq"] = test_df["kitch_sq"] / np.maximum(test_df["full_sq"].astype("float"),1)

train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]<0] = 0

train_df["ratio_kitch_sq_full_sq"].ix[train_df["ratio_kitch_sq_full_sq"]>1] = 1

test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]<0] = 0

test_df["ratio_kitch_sq_full_sq"].ix[test_df["ratio_kitch_sq_full_sq"]>1] = 1



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df.ratio_life_sq_full_sq.values, y=np.log1p(train_df.price_doc.values), size=10)

plt.ylabel('Log of Price', fontsize=12)

plt.xlabel('Ratio of living area to full area', fontsize=12)

plt.title("Joint plot on log of living price to ratio_life_sq_full_sq")

plt.show()



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df.ratio_life_sq_full_sq.values, y=np.log1p(train_df.price_doc.values), 

              kind='kde',size=10)

plt.ylabel('Log of Price', fontsize=12)

plt.xlabel('Ratio of kitchen area to living area', fontsize=12)

plt.title("Joint plot on log of living price to ratio_kitch_sq_life_sq")

plt.show()



plt.figure(figsize=(12,12))

sns.jointplot(x=train_df.ratio_life_sq_full_sq.values, y=np.log1p(train_df.price_doc.values), 

              kind='kde',size=10)

plt.ylabel('Log of Price', fontsize=12)

plt.xlabel('Ratio of kitchen area to full area', fontsize=12)

plt.title("Joint plot on log of living price to ratio_kitch_sq_full_sq")

plt.show()
# floor of the house to the total number of floors in the house #

train_df["ratio_floor_max_floor"] = train_df["floor"] / train_df["max_floor"].astype("float")

test_df["ratio_floor_max_floor"] = test_df["floor"] / test_df["max_floor"].astype("float")



# num of floor from top #

train_df["floor_from_top"] = train_df["max_floor"] - train_df["floor"]

test_df["floor_from_top"] = test_df["max_floor"] - test_df["floor"]
train_df["extra_sq"] = train_df["full_sq"] - train_df["life_sq"]

test_df["extra_sq"] = test_df["full_sq"] - test_df["life_sq"]
train_df["age_of_building"] = train_df["build_year"] - train_df["year"]

test_df["age_of_building"] = test_df["build_year"] - test_df["year"]
def add_count(df, group_col):

    grouped_df = df.groupby(group_col)["id"].aggregate("count").reset_index()

    grouped_df.columns = [group_col, "count_"+group_col]

    df = pd.merge(df, grouped_df, on=group_col, how="left")

    return df



train_df = add_count(train_df, "yearmonth")

test_df = add_count(test_df, "yearmonth")



train_df = add_count(train_df, "yearweek")

test_df = add_count(test_df, "yearweek")
train_df["ratio_preschool"] = train_df["children_preschool"] / train_df["preschool_quota"].astype("float")

test_df["ratio_preschool"] = test_df["children_preschool"] / test_df["preschool_quota"].astype("float")



train_df["ratio_school"] = train_df["children_school"] / train_df["school_quota"].astype("float")

test_df["ratio_school"] = test_df["children_school"] / test_df["school_quota"].astype("float")
train_X = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

test_X = test_df.drop(["id", "timestamp"] , axis=1)
train_y = np.log1p(train_df.price_doc.values)
val_time = 201407

dev_indices = np.where(train_X["yearmonth"]<val_time)

val_indices = np.where(train_X["yearmonth"]>=val_time)

dev_X = train_X.ix[dev_indices]

val_X = train_X.ix[val_indices]

dev_y = train_y[dev_indices]

val_y = train_y[val_indices]

print(dev_X.shape, val_X.shape)
xgb_params = {

    'eta': 0.05,

    'max_depth': 4,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'min_child_weight':1,

    'silent': 1,

    'seed':0

}



xgtrain = xgb.DMatrix(dev_X, dev_y, feature_names=dev_X.columns)

xgtest = xgb.DMatrix(val_X, val_y, feature_names=val_X.columns)

watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

num_rounds = 100 # Increase the number of rounds while running in local

model = xgb.train(xgb_params, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=5)
# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()