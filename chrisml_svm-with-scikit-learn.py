#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd

def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, log_loss

from sklearn.svm import SVC
from sklearn import decomposition

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedShuffleSplit

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')




# Swiss army knife function to organize the data

def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)           # encode species strings
    classes = list(le.classes_)                    # save column names for submission
    test_ids = test.id                             # save test ids for submission
    
    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    
    return train, labels, test, test_ids, classes

train, labels, test, test_ids, classes = encode(train, test)
train.head(1)




#sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)
#
#for train_index, test_index in sss:
#    X_train, X_test = train.values[train_index], train.values[test_index]
#    y_train, y_test = labels[train_index], labels[test_index]




pca = decomposition.PCA()
pca.fit(train)
train_t = pca.transform(train)




print(1)




# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(train_t, labels)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

#    print("Detailed classification report:")
#    print()
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    print()
#    y_true, y_pred = y_test, clf.predict(X_test)
#    print(classification_report(y_true, y_pred))
#    print()




clf.best_params_




my_svm = SVC(C=1000, kernel="linear", probability=True)
my_svm.fit(train_t, labels)

#print('****Results****')
#train_predictions = my_svm.predict(X_test)
#acc = accuracy_score(y_test, train_predictions)
#print("Accuracy: {:.4%}".format(acc))
#
#train_predictions = my_svm.predict_proba(X_test)
#ll = log_loss(y_test, train_predictions)
#print("Log Loss: {}".format(ll))
#
#
#print("="*30)




#train_t = pca.transform(train)
#my_svm.fit(train_t, labels)

test_t = pca.transform(test)
test_predictions = my_svm.predict_proba(test_t)

# Format DataFrame
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()

# Export Submission
submission.to_csv('submission.csv', index = False)

