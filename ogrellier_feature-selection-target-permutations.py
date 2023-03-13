#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from numba import jit




@jit  # for more info please visit https://numba.pydata.org/
def eval_gini(y_true, y_prob):
    """
    Original author CMPM 
    https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini




trn_df = pd.read_csv("../input/train.csv", index_col=0)
target = trn_df.target
del trn_df["target"]




clf = LGBMClassifier(boosting_type="rf",
                     num_leaves=1024,
                     max_depth=6,
                     n_estimators=500, 
                     subsample=.632,
                     colsample_bytree=.5,
                     n_jobs=2)




n_splits = 2
n_runs = 5
imp_df = np.zeros((len(trn_df.columns), n_splits * n_runs))
np.random.seed(9385610)
idx = np.arange(len(target))
for run in range(n_runs):
    # Shuffle target
    np.random.shuffle(idx)
    perm_target = target.iloc[idx]
    # Create a new split
    folds = StratifiedKFold(n_splits, True, None)
    oof = np.empty(len(trn_df))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(perm_target, perm_target)):
        trn_dat, trn_tgt = trn_df.iloc[trn_idx], perm_target.iloc[trn_idx]
        val_dat, val_tgt = trn_df.iloc[val_idx], perm_target.iloc[val_idx]
        # Train classifier
        clf.fit(trn_dat, trn_tgt)
        # Keep feature importances for this fold and run
        imp_df[:, n_splits * run + fold_] = clf.feature_importances_
        # Update OOF for gini score display
        oof[val_idx] = clf.predict_proba(val_dat)[:, 1]
        
    print("Run %2d OOF score : %.6f" % (run, eval_gini(perm_target, oof)))
    




bench_imp_df = np.zeros((len(trn_df.columns), n_splits * n_runs))
for run in range(n_runs):
    # Shuffle target AND dataset
    np.random.shuffle(idx)
    perm_target = target.iloc[idx]
    perm_data = trn_df.iloc[idx]
    
    # Create a new split
    folds = StratifiedKFold(n_splits, True, None)
    oof = np.empty(len(trn_df))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(perm_target, perm_target)):
        trn_dat, trn_tgt = perm_data.iloc[trn_idx], perm_target.iloc[trn_idx]
        val_dat, val_tgt = perm_data.iloc[val_idx], perm_target.iloc[val_idx]
        # Train classifier
        clf.fit(trn_dat, trn_tgt)
        # Keep feature importances for this fold and run
        bench_imp_df[:, n_splits * run + fold_] = clf.feature_importances_
        # Update OOF for gini score display
        oof[val_idx] = clf.predict_proba(val_dat)[:, 1]
        
    print("Run %2d OOF score : %.6f" % (run, eval_gini(perm_target, oof)))




bench_mean = bench_imp_df.mean(axis=1)
perm_mean = imp_df.mean(axis=1)

values = []
for i, f in enumerate(trn_df.columns):
    values.append((f, bench_mean[i], perm_mean[i], bench_mean[i] / perm_mean[i]))

print("%-20s | benchmark | permutation | Ratio" % "Feature")
values = sorted(values, key=lambda x: x[3])
for f, b, p, r in values[::-1]:
    print("%-20s |   %7.1f |     %7.1f |   %7.1f" 
          % (f, b, p, r))






