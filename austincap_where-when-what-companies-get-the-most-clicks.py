#!/usr/bin/env python
# coding: utf-8



filedir = '../input/'
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
from sklearn import tree
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

N_ROWS = 1000000

'''
#load documents csv's
documents_categories_df = pd.read_csv(filedir+'documents_categories.csv')
documents_entities_df = pd.read_csv(filedir+'documents_entities.csv')
documents_meta_df = pd.read_csv(filedir+'documents_meta.csv')
document_topics_df = pd.read_csv(filedir+'documents_topics.csv')
'''

#load events, clicks_train, promoted_content and page_views_sample csv's
events_df = pd.read_csv(filedir+'events.csv', nrows=N_ROWS)
promoted_content_df = pd.read_csv(filedir+'promoted_content.csv', nrows=N_ROWS)
promoted_content_df = promoted_content_df.rename(columns={'document_id': 'ad_document_id'})
clicks_train_df = pd.read_csv(filedir+'clicks_train.csv', nrows=N_ROWS)
page_views_sample_df = pd.read_csv(filedir+'page_views_sample.csv', usecols=['uuid','document_id', 'timestamp', 'traffic_source'], nrows=N_ROWS)


#merge events and clicks_train on display_id value
#in order to find what ad was clicked for each event/display_id
#this links platform and geo_location data to each ad that was clicked
events_x_clicks_train_df = pd.merge(events_df, clicks_train_df, on='display_id', how='outer').dropna()
#print(events_x_clicks_train_df)


#convert geo_location to int data
events_x_clicks_train_df['geo_location'] = events_x_clicks_train_df['geo_location'].replace('[^0-9]', '', regex=True).replace('', np.nan, regex=True)
pd.to_numeric(events_x_clicks_train_df['geo_location'])


#merge previous with page_views on uuid and document_id
#in order to find traffic source for each event/display_id
all_available_data_for_each_event = pd.merge(events_x_clicks_train_df, page_views_sample_df, on=['uuid', 'document_id','timestamp'])
all_available_data_for_each_event = pd.merge(all_available_data_for_each_event, promoted_content_df, on='ad_id', how='outer').dropna()
all_available_data_for_each_event = all_available_data_for_each_event.sort_values(by='display_id')
#print(all_available_data_for_each_event)

X = all_available_data_for_each_event[['display_id', 'document_id', 'timestamp', 'platform', 'geo_location', 'ad_id', 'traffic_source', 'ad_document_id', 'campaign_id', 'advertiser_id']]
y = all_available_data_for_each_event[['clicked']]


#rectify timestamp column and convert to hour of day clicked
pd.to_numeric(X['timestamp'])
X['true_time'] = X['timestamp']+1465876799998
X['date'] = pd.to_datetime(X['true_time'], unit='ms')
X['hour'] = X['date'].apply(lambda x: x.strftime('%H'))
X = X.drop('date', axis=1)
X = X.drop('timestamp', axis=1)
X = X.drop('true_time', axis=1)


#get platform & traffic source columns
X['platform'] = X['platform'].astype(str).convert_objects(convert_numeric=True)
X['traffic_source'] = X['traffic_source'].astype(str).convert_objects(convert_numeric=True)
X['traffic_source'] = X['traffic_source'].astype(int)

#print(X)
#print(y)
'''
all_document_ids = X.document_id.unique()
all_geo_locations = X.geo_location.unique()
all_ad_ids = X.ad_id.unique()
all_ad_document_ids = X.ad_document_id.unique()
all_campaign_ids = X.campaign_id.unique()
all_advertiser_ids = X.advertiser_id.unique()
print(len(all_geo_locations)+len(all_advertiser_ids))
'''
# Get one hot encoding of columns geo_location, platform, hour, advertiser_id, and traffic_source
# Drop column as it is now encoded, then drop irrelevant columns
X = X.join(pd.get_dummies(X['platform'], prefix='pl_'))
X = X.drop('platform', axis=1)
X = X.join(pd.get_dummies(X['geo_location'], prefix='ge_'))
X = X.drop('geo_location', axis=1)
X = X.join(pd.get_dummies(X['advertiser_id'], prefix='ad_'))
X = X.drop('advertiser_id', axis=1)
X = X.join(pd.get_dummies(X['hour'], prefix='hr_'))
X = X.drop('hour', axis=1)
X = X.join(pd.get_dummies(X['traffic_source'], prefix='ts_'))
X = X.drop('traffic_source', axis=1)

#X = X.join(pd.get_dummies(X['campaign_id']))
X = X.drop('campaign_id', axis=1)
X = X.drop('ad_id', axis=1)
X = X.drop('document_id', axis=1)
X = X.drop('ad_document_id', axis=1)
X = X.drop('display_id', axis=1)

#print(X)
#print(y)


#sklearn stuff

#clf = tree.DecisionTreeClassifier() 
#0.78

#clf = linear_model.LogisticRegression(solver='sag', multi_class='ovr') 
#0.82

#clf = linear_model.LogisticRegressionCV(Cs=100, fit_intercept=True, cv=None, 
#    dual=False, penalty='l2', 
#    solver='liblinear', tol=0.0001, max_iter=1000, refit=True, multi_class='ovr')
#0.82

#clf = linear_model.SGDClassifier() 
#0.77

clf = ensemble.RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, 
    min_samples_split=20, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
    max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-02, bootstrap=True, 
    class_weight=None)
#0.82

#tree.export_graphviz(clf, out_file='tree.dot')  

#clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
#0.82

#clf = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, 
#    tol=0.00001, C=1.0, multi_class='crammer_singer', fit_intercept=False, 
#    intercept_scaling=1, max_iter=2000)
#0.81

#clf = svm.NuSVC(nu=0.5, kernel='rbf', degree=3, gamma='auto', coef0=0.1, shrinking=True, 
#    probability=False, tol=0.0001, cache_size=20,
#    max_iter=1000, decision_function_shape='ovr', random_state=None)
#0.60

#clf = linear_model.PassiveAggressiveClassifier(C=1.0, fit_intercept=True, n_iter=5, 
#    shuffle=True, verbose=0, loss='hinge', n_jobs=1, random_state=None, 
#    warm_start=False, class_weight=None)
#0.81

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
clf.fit(X_train.as_matrix(), y_train.as_matrix())
print(clf.score(X_test, y_test))
np.set_printoptions(threshold=np.nan)
#print(clf.predict(X_test))
#print(y_test)
print(metrics.f1_score(y_test, clf.predict(X_test)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
clf.fit(X_train.as_matrix(), y_train.as_matrix())
print(clf.score(X_test, y_test))
print(metrics.f1_score(y_test, clf.predict(X_test)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)
clf.fit(X_train.as_matrix(), y_train.as_matrix())
print(clf.score(X_test, y_test))
print(metrics.f1_score(y_test, clf.predict(X_test)))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()
    return plt

title = "Learning Curves (Random Forest)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cvx = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
y = y.as_matrix().ravel()
plot_learning_curve(clf, title, X.as_matrix(), y, ylim=(0.7, 1.01), cv=cvx, n_jobs=4)

#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#estimator = SVC(gamma=0.001)
#plot_learning_curve(estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)




# I seem to get an accuracy of about 0.82 no matter which algorithm I use.

