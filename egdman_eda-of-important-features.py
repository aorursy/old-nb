#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


feature_names = ['L3_S38_F3960', 'L3_S33_F3865', 'L3_S38_F3956', 'L3_S33_F3857',
       'L3_S29_F3321', 'L1_S24_F1846', 'L3_S32_F3850', 'L3_S29_F3354',
       'L3_S29_F3324', 'L3_S35_F3889', 'L0_S1_F28', 'L1_S24_F1844',
       'L3_S29_F3376', 'L0_S0_F22', 'L3_S33_F3859', 'L3_S38_F3952', 
       'L3_S30_F3754', 'L2_S26_F3113', 'L3_S30_F3759', 'L0_S5_F114']

# my features
feature_names = ['L3_S33_F3859', 'L3_S33_F3865', 'L3_S33_F3857', 'L1_S24_F1846',
                 'L3_S32_F3850', 'L3_S29_F3351', 'L3_S30_F3744', 'L1_S24_F1723',
                 'L3_S30_F3759', 'L3_S29_F3407', 'L1_S24_F1844', 'L3_S36_F3920',
                 'L3_S30_F3809', 'L3_S30_F3704', 'L3_S30_F3769', 'L3_S30_F3754',
                 'L3_S29_F3339', 'L0_S0_F16', 'L3_S30_F3829', 'L3_S29_F3348']




numeric_cols = pd.read_csv("../input/train_numeric.csv", nrows = 1).columns.values
imp_idxs = [np.argwhere(feature_name == numeric_cols)[0][0] for feature_name in feature_names]
train = pd.read_csv("../input/train_numeric.csv", 
                index_col = 0, header = 0, usecols = [0, len(numeric_cols) - 1] + imp_idxs)
train = train[feature_names + ['Response']]




X_neg, X_pos = train[train['Response'] == 0].iloc[:, :-1], train[train['Response']==1].iloc[:, :-1]




BATCH_SIZE = 5
train_batch =[pd.melt(train[train.columns[batch: batch + BATCH_SIZE].append(np.array(['Response']))], 
                      id_vars = 'Response', value_vars = feature_names[batch: batch + BATCH_SIZE])
              for batch in list(range(0, train.shape[1] - 1, BATCH_SIZE))]




FIGSIZE = (12,16)
_, axs = plt.subplots(len(train_batch), figsize = FIGSIZE)
plt.suptitle('Univariate distributions')
for data, ax in zip(train_batch, axs):
    sns.violinplot(x = 'variable',  y = 'value', hue = 'Response', data = data, ax = ax, split =True)




non_missing = pd.DataFrame(pd.concat([(X_neg.count()/X_neg.shape[0]).to_frame('negative samples'),
                                      (X_pos.count()/X_pos.shape[0]).to_frame('positive samples'),  
                                      ], 
                       axis = 1))
non_missing_sort = non_missing.sort_values(['negative samples'])
non_missing_sort.plot.barh(title = 'Proportion of non-missing values', figsize = FIGSIZE)
plt.gca().invert_yaxis()




FIGSIZE = (13,4)
_, (ax1, ax2) = plt.subplots(1,2, figsize = FIGSIZE)
MIN_PERIODS = 100

triang_mask = np.zeros((X_pos.shape[1], X_pos.shape[1]))
triang_mask[np.triu_indices_from(triang_mask)] = True

ax1.set_title('Negative Class')
sns.heatmap(X_neg.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax1)

ax2.set_title('Positive Class')
sns.heatmap(X_pos.corr(min_periods = MIN_PERIODS), mask = triang_mask, square=True,  ax = ax2)




sns.heatmap(X_pos.corr(min_periods = MIN_PERIODS) -X_neg.corr(min_periods = MIN_PERIODS), 
             mask = triang_mask, square=True)




nan_pos, nan_neg = np.isnan(X_pos), np.isnan(X_neg)

triang_mask = np.zeros((X_pos.shape[1], X_pos.shape[1]))
triang_mask[np.triu_indices_from(triang_mask)] = True

FIGSIZE = (13,4)
_, (ax1, ax2) = plt.subplots(1,2, figsize = FIGSIZE)
MIN_PERIODS = 100

ax1.set_title('Negative Class')
sns.heatmap(nan_neg.corr(),   square=True, mask = triang_mask, ax = ax1)

ax2.set_title('Positive Class')
sns.heatmap(nan_pos.corr(), square=True, mask = triang_mask,  ax = ax2)




sns.heatmap(nan_neg.corr() - nan_pos.corr(), mask = triang_mask, square=True)




# print feature pairs with highest correlation
triang_mask = np.ones((X_pos.shape[1], X_pos.shape[1]))
triang_mask[np.triu_indices_from(triang_mask)] = 0

diff_nan_corr = np.multiply((nan_neg.corr() - nan_pos.corr()).values, triang_mask)
sorted_idx = np.dstack(np.unravel_index(np.argsort(diff_nan_corr.ravel()), diff_nan_corr.shape))[0, ::-1]

pairs = []
for idx in sorted_idx[:36]:
    pairs.append((feature_names[idx[0]], feature_names[idx[1]]))
    print("{} : {}".format(pairs[-1][0], pairs[-1][1]))
    
# print(pairs)

