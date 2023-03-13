#!/usr/bin/env python
# coding: utf-8



from sklearn.metrics import fbeta_score
y_true = [0, 1, 2, 0, 1, 2] #Sample data
y_pred = [0, 2, 1, 0, 0, 1] #Sample Predication
labels = ['agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
          'blow_down', 'clear', 'cloudy', 'conventional_mine', 'cultivation',
          'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 
          'selective_logging', 'slash_burn', 'water']




print ("Calculate metrics globally by counting the total true positives false negatives and false positives:", 
       fbeta_score(y_true, y_pred, average='macro',   labels=None, beta=0.5))
print ("Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.:", 
       fbeta_score(y_true, y_pred, average='micro',   labels=None, beta=0.5))
print ("Calculate metrics for each label, and find their average, weighted         by support (the number of true instances for each label):", 
       fbeta_score(y_true, y_pred, average='weighted',labels=None, beta=0.5))
print ("print onlye the array of predication:",
       fbeta_score(y_true, y_pred, average=None, labels=None, beta=0.5))









#FBeta_Score(y_true, y_pred, positive = NULL, beta = 1)
# for R you can call the package from github which is by Yanachen abd include the package.
# https://github.com/yanyachen/MLmetrics
require(yanyachen/MLmetrics)

tr <- c(0, 1, 2, 0, 1, 2)
pr <- c(0, 2, 1, 0, 0, 1)
FBeta_Score(y_pred = pr, y_true = tr, positive = "0", beta = 2)
FBeta_Score(y_pred = pr, y_true = tr, positive = "1", beta = 2)

