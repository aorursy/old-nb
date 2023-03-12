
import matplotlib.pylab as plt

import numpy as np

import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

from scipy import stats

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import StratifiedKFold

import itertools as itertools

from sklearn.metrics import log_loss



def get_skf_indexes(df, target, kfold=4):

    X = df.values

    y = df[target].values

    skf = StratifiedKFold(n_splits=4);

    skf.get_n_splits(X, y);

    indexes = [[],[]]

    for train_index, test_index in skf.split(X, y):

        indexes[0].append(train_index) # Training indexes

        indexes[1].append(test_index) # test indexes

    return indexes





def get_lr_perf(df_train, df_test, feature='__to_check', target='response', n_quantile=20):

    results = {}

    # Inputs

    xtrain = df_train[feature].values.reshape(-1,1)

    ytrain = df_train[target].values

    xtest = df_test[feature].values.reshape(-1,1)

    ytest = df_test[target].values

    # Evaluation as a single feature

    lr = LogisticRegression()

    lr.fit(xtrain, ytrain);

    yptrain = lr.predict_proba(xtrain)

    yptest = lr.predict_proba(xtest)

    results['train.num'] = np.round(log_loss(ytrain, yptrain), 6)

    results['test.num'] = np.round(log_loss(ytest, yptest), 6)

    # Evaluation as a categorical feature using quantile buckets

    bins = np.unique(np.percentile(xtrain, np.arange(n_quantile, 100, n_quantile)))

    xtrainq = np.digitize(xtrain, bins)

    xtestq = np.digitize(xtest, bins)

    lb = LabelBinarizer()

    x1 = lb.fit_transform(xtrainq)

    x2 = lb.transform(xtestq)

    lr.fit(x1, ytrain);

    yptrain = lr.predict_proba(x1)

    yptest = lr.predict_proba(x2)

    results['train.cat'] = np.round(log_loss(ytrain, yptrain), 6)

    results['test.cat'] = np.round(log_loss(ytest, yptest), 6)

    return results
df = pd.read_json('../input/train.json')

df['response'] = "0"

df.loc[df.interest_level=='medium', 'response'] = "1"

df.loc[df.interest_level=='high', 'response'] = "2"
# Parameters to check

AA = (0.1, 0.5, 1, 2)

CC = ((0, 4), (0, 3), (1, 4), (1, 3), (0, 2))

DD = ((0, 3), (0, 2), (1, 3), (1, 2))

BB = (0, 0.25, 0.5, 1, 2)

# Reduced set of parameters to run here

AA = (0.5, 1, 2)

CC = ((0, 4), (0, 3), (1, 4), (1, 3))

DD = ((0, 3), (0, 2))

BB = (0.25, 0.5, 1)

# Stratified kfold

idx_train, idx_test = get_skf_indexes(df, 'response', kfold=2) # kfold=4, set to 2 to quickly run here

# Get results

Y = pd.DataFrame()

for iper, (i_train, i_test) in enumerate(zip(idx_train, idx_test)):

    print(iper)

    df_train = df.iloc[i_train, :].copy()

    df_test = df.iloc[i_test, :].copy()

    # For each parameter combination

    for A, C, D, B in itertools.product(AA, CC, DD, BB):

        df_train['__to_check'] = (df_train.price / (A + df_train.bedrooms.clip(C[0], C[1]) + B*df_train.bathrooms.clip(D[0], D[1]))).values

        df_test['__to_check'] = (df_test.price / (A + df_test.bedrooms.clip(C[0], C[1]) + B*df_test.bathrooms.clip(D[0], D[1]))).values

        results = get_lr_perf(df_train, df_test, feature='__to_check', target='response', n_quantile=20)

        results.update({'fold': iper, 'params': {'A':A, 'B': B, 'C': C, 'D':D}})

        Y =  Y.append(pd.DataFrame(pd.Series(results)).transpose())

for i in ['train.cat', 'train.num', 'test.cat', 'test.num']:

    Y[i] = Y[i].astype(float)
Y.sort_values('test.cat')
df['half_bathrooms'] = ((np.round(df.bathrooms) - df.bathrooms)!=0).astype(float) # Half bathrooms? 1.5, 2.5, 3.5...

df['bathrooms'] = df.bathrooms.clip(0,4) # Reduce outlier effects
# Build two models with and without 'half_bathrooms' feature

formula1 = 'response ~ bathrooms'

formula2 = 'response ~ bathrooms + half_bathrooms'

model1 = smf.glm(formula=formula1, data=df, family=sm.families.Binomial())

model2 = smf.glm(formula=formula2, data=df, family=sm.families.Binomial())

result1 = model1.fit()

result2 = model2.fit()

# Likelihood ratio test

llf_1 = result1.llf

llf_2 = result2.llf

df_1 = result1.df_resid 

df_2 = result2.df_resid 

lrdf = (df_1 - df_2)

lrstat = -2*(llf_1 - llf_2)

lr_pvalue = stats.chi2.sf(lrstat, df=lrdf)

# Print results

print(formula1)

print(result1.summary())

print(formula2)

print(result2.summary())

print('Likelihood ratio test', lr_pvalue)
x = pd.crosstab(df.bathrooms, df.interest_level)[['low', 'medium', 'high']]

x.div(x.sum(1), 0).plot(kind='bar', color=['red', 'yellow', 'green'], stacked=True);