#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(train_series=None,
                  test_series=None,
                  target=None,
                  noise_level=0):
    assert len(train_series) == len(target)
    assert train_series.name == test_series.name

    temp = pd.concat([train_series, target], axis=1)
    # Compute target mean
    aggregated_values = temp.groupby(by=train_series.name)[target.name].agg(["mean", "count", np.std])
    total_std = np.std(target)
    aggregated_values["std"].fillna(total_std, inplace=True)

    # Compute smoothing
    smoothing_component = aggregated_values["count"] * total_std ** 2
    smoothing = smoothing_component / (aggregated_values["std"] ** 2 + smoothing_component)

    # Apply average function to all target data
    mean_total = target.mean()
    mean_values = mean_total * (1 - smoothing) + aggregated_values["mean"] * smoothing

    mean_values_dict = mean_values.rank(axis=0, method='first').to_dict()

    train_columns = train_series.replace(mean_values_dict).fillna(mean_total)
    test_columns = test_series.replace(mean_values_dict).fillna(mean_total)
    
    return add_noise(train_columns, noise_level), add_noise(test_columns, noise_level)




# reading data
trn_df = pd.read_csv("../input/train.csv", index_col=0)
sub_df = pd.read_csv("../input/test.csv", index_col=0)

# Target encode ps_car_11_cat
trn, sub = target_encode(trn_df["ps_car_11_cat"], 
                         sub_df["ps_car_11_cat"], 
                         target=trn_df.target,
                         noise_level=0.01)
trn.head(10)




import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

i=0
for f in trn_df.columns:
    if "_cat" in f:
        trn, sub = target_encode(trn_df[f], 
                         sub_df[f], 
                         target=trn_df.target,
                         noise_level=0)

        plt.figure(i)
        i+= 1
        plt.scatter(trn_df[f], trn)
        plt.xlabel(f + " category values")
        




from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
f_cats = [f for f in trn_df.columns if "_cat" in f]
print("%20s   %20s | %20s" % ("", "Raw Categories", "Encoded Categories"))
for f in f_cats:
    print("%-20s : " % f, end="")
    e_scores = []
    f_scores = []
    for trn_idx, val_idx in folds.split(trn_df.values, trn_df.target.values):
        trn_f, trn_tgt = trn_df[f].iloc[trn_idx], trn_df.target.iloc[trn_idx]
        val_f, val_tgt = trn_df[f].iloc[trn_idx], trn_df.target.iloc[trn_idx]
        trn_tf, val_tf = target_encode(train_series=trn_f, 
                                       test_series=val_f, 
                                       target=trn_tgt,
                                       noise_level=0.01)
        f_scores.append(max(roc_auc_score(val_tgt, val_f), 1 - roc_auc_score(val_tgt, val_f)))
        e_scores.append(roc_auc_score(val_tgt, val_tf))
    print(" %.6f + %.6f | %6f + %.6f" 
          % (np.mean(f_scores), np.std(f_scores), np.mean(e_scores), np.std(e_scores)))

