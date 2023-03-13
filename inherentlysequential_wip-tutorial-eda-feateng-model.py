#!/usr/bin/env python
# coding: utf-8



from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  # for better interative experience

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'  # for high resolution plots")




import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
pd.options.display.max_columns = 100

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('ticks')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))




get_ipython().run_cell_magic('time', '', 'train = pd.read_csv("../input/train.csv", index_col="id", engine=\'python\',\n                    parse_dates=[\'pickup_datetime\', \'dropoff_datetime\'],\n                    infer_datetime_format=True)\n\ntest = pd.read_csv("../input/test.csv", index_col="id", engine=\'python\',\n                   parse_dates=[\'pickup_datetime\'],\n                   infer_datetime_format=True)')




# before changing the types, make sure train and test have the same categories. Otherwise,
# we'll get into trouble at test time!
print(f"train and test vendor_id unique values are the same:         {train.vendor_id.unique().sort() == test.vendor_id.unique().sort()}")
print(f"train and test store_and_fwd_flag unique values are the same:         {train.store_and_fwd_flag.unique().sort() == test.store_and_fwd_flag.unique().sort()}")

train['vendor_id'] = train['vendor_id'].astype('category')
train['store_and_fwd_flag'] = train['store_and_fwd_flag'].astype('category')

test['vendor_id'] = test['vendor_id'].astype('category')
test['store_and_fwd_flag'] = test['store_and_fwd_flag'].astype('category')




print(f"train shape: {train.shape}")
print("=======================================================")
print(f"test shape: {test.shape}")
print("=======================================================")
print("train view:")
train.head()
print("test view:")
test.head()
print("train/test columns difference:")
train.columns.difference(test.columns)
print("=======================================================")
print("train info:")
train.info()
print("=======================================================")
print("test info:")
test.info()
print("=======================================================")
print(f"train feature descriptions:")
train.describe()
print(f"test feature descriptions:")
test.describe()
print("=======================================================")
print(f"train/test overlapping index: {train.index.intersection(test.index)}")




# Do NOT run! it'll take a long time
#%%time
#train_uniq = train.apply(lambda s: s.nunique())
#test_uniq = test.apply(lambda s: s.nunique())

#print(f"Number of unique values per column in train: \n {pd.DataFrame(train_uniq).T}")
#print(f"Number of unique values per column in test: \n {pd.DataFrame(test_uniq).T}")




target = train.trip_duration

print(f"trip_duration \n       min: {target.min()} \n       max: {target.max()} \n       mode: {target.mode()[0]} \n       mean: {target.mean()} \n       median: {target.median()} \n       1% quantile: {target.quantile(q=0.01)} \n       99% quantile: {target.quantile(q=0.99)}")




train_pure = train.query(f"{target.quantile(q=0.01)} <=                             trip_duration <= {target.quantile(q=0.99)}")
target_pure = train_pure.trip_duration 
sns.distplot(target_pure)
plt.xlabel("trip duration")
plt.title("Sample distribution of tip duration");




sns.distplot(np.log(target_pure))
plt.xlabel("log trip duration")
plt.title("Sample distribution of logarithm of tip duration");




from scipy.stats import shapiro, anderson, kstest, normaltest

target_pure_log = np.log(target_pure)  # target values are already positive

print(f"p-value for Shapiro-Wilk test: {shapiro(target_pure_log)[1]}")

print(f"A 2-sided chi squared probability for the hypothesis test:         {normaltest(target_pure_log).pvalue}")

print(f"p-value for the Kolmogorov-Smirnov test: {kstest(target_pure, cdf='norm').pvalue}")

print(f"Anderson-Darling normality test: \n         {anderson(target_pure_log, dist='norm')}")




train_sample = train_pure[:10000]
sns.pairplot(train_sample, hue='vendor_id');




train_corr = train_pure.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(train_corr, annot=True)
plt.title("Correlation plot");




print(f"vendor_id value counts: \n {train_pure.vendor_id.value_counts()}")
print(f"store_and_fwd_flag value counts: \n {train_pure.store_and_fwd_flag.value_counts()}")

fig, ax = plt.subplots(1,2, figsize=(15, 7))
sns.countplot(data=train_pure, x='vendor_id', ax=ax[0])
sns.countplot(data=train_pure, x='store_and_fwd_flag', ax=ax[1])
plt.show();




plt.figure(figsize=(12, 10))
ax = sns.boxplot(x='vendor_id', y='trip_duration', data=train_pure[:1000])
ax = sns.swarmplot(x='vendor_id', y='trip_duration', data=train_pure[:1000], color='k');
plt.xlabel("vendor id")
plt.ylabel("trip duration")
plt.show();




g = sns.FacetGrid(train_pure, row='vendor_id', hue='store_and_fwd_flag',
                  aspect=3, size=2.5, margin_titles=True)
g.map(sns.kdeplot, 'trip_duration', shade=True).add_legend()
for ax in g.axes.flat:
    ax.yaxis.set_visible(False)
sns.despine(left=True);




print(f"unique values: {train_pure.passenger_count.unique()}")
print(f"number of unique values: {train_pure.passenger_count.nunique()}")

sns.barplot(x="passenger_count", y="trip_duration", data=train_pure, estimator=np.median);
plt.xlabel("passenger count")
plt.ylabel("median of trip duration (seconds)")
plt.title("Median of trip duration barplot over passenger count");




plt.figure(figsize=(16, 12))
g = sns.FacetGrid(train_pure, col='passenger_count',
                  col_wrap=3, hue='vendor_id',
                  aspect=1, size=2, margin_titles=True)
g.map(sns.kdeplot,  'trip_duration', shade=True).add_legend()
for ax in g.axes.flat:
    ax.yaxis.set_visible(False)
sns.despine(left=True);




train_trip_vendor = train_pure.pivot_table('trip_duration',
                                           index='passenger_count', 
                                           columns='vendor_id',
                                           aggfunc='median',
                                           margins='All')
train_trip_vendor




train_trip_vendor_flag = train_pure.pivot_table('trip_duration',
                                                index='passenger_count', 
                                                columns=['vendor_id',
                                                         'store_and_fwd_flag'],
                                                aggfunc='median')

train_trip_vendor_flag

train_trip_vendor_flag.plot()
plt.xlabel("passenger count")
plt.ylabel("median of trip_duration (seconds)")
plt.show()




pickup_hour = pd.qcut(train_pure['pickup_datetime'].dt.hour, q=[0, .25, .5, .75, 1.])
train_trip_vendor_flag = train_pure.pivot_table('trip_duration',
                                                index='passenger_count', 
                                                columns=['vendor_id',
                                                         'store_and_fwd_flag',
                                                         pickup_hour],
                                                aggfunc='median')

train_trip_vendor_flag




train_pure['pickup_date'] = train_pure['pickup_datetime'].dt.date
train_pure['pickup_time'] = train_pure['pickup_datetime'].dt.time
train_pure['pickup_month'] = train_pure['pickup_datetime'].dt.month
train_pure['pickup_day'] = train_pure['pickup_datetime'].dt.day
train_pure['pickup_hour'] = train_pure['pickup_datetime'].dt.hour




train_pkdt = train_pure.set_index('pickup_datetime')

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
train_pkdt['trip_duration'].resample('M').median().plot(style='-', ax=ax[0])
train_pkdt['trip_duration'].resample('W').median().plot(style='--', ax=ax[0])
train_pkdt['trip_duration'].resample('D').median().plot(style=':', ax=ax[0])
train_pkdt['trip_duration'].resample('H').median().plot(alpha=0.3, color='k', ax=ax[0])
ax[0].set_xlabel('')
ax[0].set_ylabel('median')
ax[0].set_title('Median trip durations over difference time intervals')
ax[0].legend(['Monthly', 'Weekly', 'Daily', 'Hourly'], loc='upper right')

train_pkdt['trip_duration'].resample('D').count().plot(ax=ax[1])
ax[1].set_xlabel('pickup datetime')
ax[1].set_ylabel('count')

fig.show();




train_pkdt['trip_duration'].resample('M').median().plot(style='-')
train_pkdt['trip_duration'].resample('W').median().plot(style='--')
train_pkdt['trip_duration'].resample('D').median().plot(style=':')
plt.xlabel('pickup datetime')
plt.title('Median trip durations over difference time intervals')
plt.legend(['Monthly', 'Weekly', 'Daily'], loc='best')
plt.show();



















from fbprophet import Prophet




df_train = train_pkdt['trip_duration'].reset_index()
df_train.rename(columns={'pickup_datetime': 'ds', 'trip_duration': 'y'}, inplace=True)

df_test = test['pickup_datetime'].rename(columns={'pickup_datetime': 'ds'})




get_ipython().run_cell_magic('time', '', 'prophet = Prophet(interval_width=0.95)\nprophet.fit(df_train.loc[:1000, :])')









pred = prophet.predict(df_test.loc[:1000])




df_test.head()






