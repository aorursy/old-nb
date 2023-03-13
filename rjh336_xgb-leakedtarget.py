#!/usr/bin/env python
# coding: utf-8



get_ipython().system('ls ..')




import pandas as pd
import numpy as np
import gc
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df_train = pd.read_csv('../input/train.csv').sample(n=3000000) # transactions
df_members = pd.read_csv('../input/members.csv') # members
df_songs = pd.read_csv('../input/songs.csv') 
song_extra = pd.read_csv('../input/song_extra_info.csv')
df_songs = df_songs.merge(song_extra,how='left',on='song_id') # merge of songs and song attributes




# Merge songs, members to the transactions data

df_train = pd.merge(pd.merge(df_train, df_members, how='left', on='msno'), df_songs, how='left', on='song_id')
del df_members, df_songs, song_extra; gc.collect();




# Take a look at null counts and dtypes
df_train.info(null_counts=True)




# Convert to categorical from numerics:

df_train['language'] = df_train['language'].apply(str)
df_train['city'] = df_train['city'].apply(str)
df_train['registered_via'] = df_train['registered_via'].apply(str)
df_train['genre_ids'] = df_train['genre_ids'].apply(str)




listens = df_train[['song_id', 'target']].groupby(['song_id']).agg(['mean','count']).reset_index()
listens.columns = listens.columns.droplevel()
# Because target is binary (1,0) we can take the mean of occurences to get probability
listens.columns = ['song_id','mean','count']
listens['replay_prob'] = listens['mean'] * listens['count']
df_train = df_train.merge(listens[['song_id','replay_prob']], how='left', on='song_id')




# Get the year of song release from isrc code
def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return 1900 + int(isrc[5:7])
        else:
            return 2000 + int(isrc[5:7])
    else:
        return np.nan
        
df_train['song_year'] = pd.to_numeric(df_train.isrc.apply(isrc_to_year))
df_train.song_year.hist(bins=70)




# Create popularity (counts) for appearances of songs, genres, artists, and languages

popularity_features = ['song_id','genre_ids','artist_name','language']
for feat in popularity_features:
    pop_df = pd.DataFrame(df_train[feat].value_counts()).reset_index().rename(
        columns={'index': feat, feat: feat+'_popularity'})
    df_train = df_train.merge(pop_df, how='left', on=feat)




fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(22,18))
sns.barplot(x="genre_ids", y="genre_ids_popularity", data=df_train[df_train.genre_ids_popularity > 15], ax=ax1)
sns.barplot(x="language", y="language_popularity", data=df_train, ax=ax2)
plt.setp(ax1.get_xticklabels(), rotation=90)




# Create user song consumption feature ('msno_volume')
volume = pd.DataFrame(df_train.msno.value_counts()).reset_index().rename(
    columns={'index': 'msno', 'msno': 'msno_volume'})
df_train = df_train.merge(volume, how='left', on='msno')
del volume
gc.collect()




df_train.columns




df_train.msno_volume.hist(bins=40)
plt.title("Song Listens across KKBOX Users")




# get membership length
df_train.registration_init_time = pd.to_datetime(df_train.registration_init_time,format="%Y%m%d")
df_train.expiration_date = pd.to_datetime(df_train.expiration_date,format="%Y%m%d")
df_train['membership_days'] = (df_train.expiration_date - df_train.registration_init_time)
df_train['membership_days'] = (df_train.membership_days/np.timedelta64(1, 'D')).astype('int')
df_train['membership_days'] = df_train.membership_days.clip(lower=0)




df_train.membership_days.hist(bins=40)
plt.title("Membership Days across KKBOX Users")




df_train.bd.hist(bins=40)




# There are also some obvious instances of missing data:
df_train.bd[(df_train.bd > 100) | (df_train.bd < 5)].value_counts()




# Impute missing ages
age_mean = df_train[(df_train.bd > 0)&(df_train.bd < 100)].bd.mean()
df_train.bd = df_train.bd.apply(lambda x: age_mean if x<=0 or x>100 else x)




# That looks better
df_train.bd.hist(bins=40)




exlcude_cols = ['genre_ids', 'msno', 'song_id', 'isrc', 'name', 'composer', 'lyricist', 'artist_name','language']
cat_cols = [x for x in df_train.select_dtypes(include=['object']).columns if x not in exlcude_cols]
cat_cols




fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6, figsize=(20,55))
axes=(ax1, ax2, ax3, ax4, ax5, ax6)
for i, col in enumerate(cat_cols):
    vc = df_train[col].value_counts().reset_index().rename(columns={col:'count','index':col})

    sns.barplot(x="count", y=col, data=vc, ax=axes[i], orient='h')




df_train[['genre_ids','composer','lyricist','artist_name']].tail()




def count_vals(x):
    # count number of values separated by '|'
    if type(x) != str:
        return 1
    else:
        return 1 + x.count('|')
    
df_train['number_of_genres'] = df_train['genre_ids'].apply(count_vals)
df_train['number_of_composers'] = df_train['composer'].apply(count_vals)
df_train['number_of_lyricists'] = df_train['lyricist'].apply(count_vals)
df_train['number_of_artists'] = df_train['artist_name'].apply(count_vals)




# merge user tendencies by artist, genre, and language
song_features = ['artist_name', 'genre_ids', 'language']
for feature in song_features:
    listens = feature+'_listens'
    listens_pct = listens+'_pct'
    listens_by_feat = df_train.groupby(['msno', feature]).                                count()['song_id'].                                reset_index().                                rename(columns={'song_id':listens})
    df_train = df_train.merge(listens_by_feat,how='left',on=['msno',feature])
    df_train[listens_pct] = df_train[listens] / df_train['msno_volume']
    assert df_train[df_train[listens_pct] > 1].empty and df_train[df_train[listens_pct] < 0].empty
    df_train.drop([listens], axis=1, inplace=True)




mask = ['replay_prob', 'song_year','song_id_popularity',
       'genre_ids_popularity', 'artist_name_popularity', 'language_popularity',
       'msno_volume', 'membership_days', 'number_of_genres',
       'number_of_composers', 'number_of_lyricists', 'number_of_artists',
       'artist_name_listens_pct', 'genre_ids_listens_pct', 'language_listens_pct']

df_corr = df_train[mask].fillna(0)




df_corr.corr()




# Create dummies for the categorical features and merge back to transactions
for c in cat_cols:
    split = pd.get_dummies(df_train[c])
    new_names = {i : c+str(i) for i in split.columns}
    split.rename(columns = new_names, inplace=True)
    df_train = df_train.merge(split,how='left',left_index=True,right_index=True)




df_train.shape




from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_train, test_size=.15)




# Calculate the prob(repeat) per user for a given cross feature to the train set
# Then merge probs to test set
crosses = ['artist_name', 'language', 'genre_ids']

for feat in crosses:
    cross_col = "X_" + 'msno' + '_' + feat
    df_train[cross_col] = df_train['msno'] + df_train[feat]
    df_test[cross_col] = df_test['msno'] + df_test[feat]
    # Because target is binary [0,1], I can take the mean to get a
    # probability of the target for the given cross feature
    target_total = df_train.groupby(cross_col).mean()['target'].reset_index()
    # Check to make sure probabilities are between 0 and 1
    assert target_total[target_total.target > 1].empty and target_total[target_total.target < 0].empty
    # Create prob columns in train and merge to test
    target_total = target_total.rename(columns={'target': 'target_prob_' + feat})
    df_train = df_train.merge(target_total, how='left', on=cross_col)
    df_test = df_test.merge(target_total, how='left', on=cross_col)




y_test = df_test.target
X_test = df_test.loc[:,df_train.columns != 'target']

y_train = df_train.target
X_train = df_train.loc[:,df_train.columns != 'target']




from sklearn.preprocessing import scale

def preprocess(df_model):
    # leave numerics only
    df_model = df_model.select_dtypes(include=['int64', 'int32', 'float64', 'uint8']).fillna(0)
    df_scaled = pd.DataFrame(scale(df_model), columns=df_model.columns)
    return df_scaled




X_test = preprocess(X_test)
X_train = preprocess(X_train)

del df_train, df_test; gc.collect();

# Classes are fairly balanced
print("\nCLASS BALANCE: ")
print(y_train.value_counts() / len(y_train))
print("\nSIZES: ")
print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)




X = pd.concat([X_train, X_test]).fillna(0)
y = pd.concat([y_train, y_test]).fillna(0)




from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold
from sklearn.metrics import accuracy_score, roc_auc_score




models = [KNeighborsClassifier(), 
               LogisticRegression(),
               SGDClassifier(), 
               RandomForestClassifier(), 
               XGBClassifier()]




kf = KFold(n_splits=10, shuffle=True, random_state=42)
step = 100
for est in models:
    print('\n',est.get_params)
    cv_roc = cross_val_score(est, X[::step], y[::step], scoring='roc_auc', cv=kf, n_jobs=-1)
    cv_acc = cross_val_score(est, X[::step], y[::step], scoring='accuracy', cv=kf, n_jobs=-1)
    print('CV ROC AUC:  ', cv_roc, np.mean(cv_roc))
    print('CV Accuracy: ', cv_acc, np.mean(cv_acc))




def test_train_scores(model):
    predictions = model.predict(X_train[::step])
    auc = roc_auc_score(y_train[::step], predictions)
    accuracy = accuracy_score(y_train[::step], predictions)
    print("ROC TRAIN AUC   : ", auc)
    print("ACCURACY  : ", accuracy)

    predictions = model.predict(X_test[::step])
    auc = roc_auc_score(y_test[::step], predictions)
    accuracy = accuracy_score(y_test[::step], predictions)
    print("ROC TEST  AUC   : ", auc)
    print("ACCURACY  : ", accuracy)

def plot_learning_curves(model):
    # retrieve performance metrics
    results = model.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)

    # plot roc auc
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x_axis, results['validation_0']['auc'], label='Train')
    ax.plot(x_axis, results['validation_1']['auc'], label='Test')
    ax.legend()
    plt.ylabel('ROC AUC')
    plt.xlabel('Number of Epochs')
    plt.title('XGBoost ROC AUC')
    plt.show()

    # plot classification error
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x_axis, results['validation_0']['error'], label='Train')
    ax.plot(x_axis, results['validation_1']['error'], label='Test')
    ax.legend()
    plt.ylabel('Error')
    plt.xlabel('Number of Epochs')
    plt.title('XGBoost Classification Error')
    plt.show()




from scipy.stats import randint as sp_randint

params = {'max_depth': [3,4,5,6],
          'gamma':[15,20,30],
          'subsample':[.1,.8],
          'colsample_bytree':[.1,.8],
          'learning_rate':sp_randint(1e-10,9e-2)
          }

model = RandomizedSearchCV(XGBClassifier(n_estimators=3000), 
                     param_distributions=params, n_jobs=1, scoring='roc_auc')




model = XGBClassifier(n_estimators=1500, learning_rate=.000000001, max_depth=3,
                      gamma=5, subsample=.2, colsample_bytree=.125)




get_ipython().run_cell_magic('time', '', "step = 5\neval_set = [(X_train[::step], y_train[::step]), (X_test[::step], y_test[::step])]\nmodel.fit(X_train[::step], y_train[::step], \n          eval_metric=['auc','error'],eval_set=eval_set, verbose=False)")




test_train_scores(model)




plot_learning_curves(model)




feat_imp = sorted(list(zip(X_train.columns, model.feature_importances_)),key=lambda tup: tup[1],reverse=True)
feat_imp = pd.DataFrame(feat_imp)[:20]
sns.set(style="whitegrid")
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))
sns.set_color_codes("pastel")
fi = sns.barplot(x=1, y=0, data=feat_imp, color="b")
ax.set(xlabel="Feature Importance Score")
sns.despine(left=True, bottom=True)






