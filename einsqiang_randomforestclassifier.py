import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.metrics import roc_auc_score
from collections import defaultdict

train_data = pd.read_csv("../input/train.csv", index_col=0)
test_data = pd.read_csv("../input/test.csv", index_col=0)

print(train_data.shape)
print(test_data.shape)
X = train_data.iloc[: , : -1]
y = train_data.TARGET
X['n0'] = (X == 0).sum(axis=1)
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif, chi2
from sklearn.preprocessing import Binarizer, scale
p = 30

X_bin = Binarizer().fit_transform(scale(X))
selectChi2 = SelectPercentile(chi2, percentile=p).fit(X_bin, y)
selectF_classif = SelectPercentile(f_classif, percentile=p).fit(X, y)

chi2_selected = selectChi2.get_support()
chi2_selected_features = [
    f for i, f in enumerate(X.columns) if chi2_selected[i]]
print('Chi2 selected {} features {}.'.format(
    chi2_selected.sum(), chi2_selected_features))

f_classif_selected = selectF_classif.get_support()
f_classif_selected_features = [
    f for i, f in enumerate(X.columns) if f_classif_selected[i]]
print('F_classif selected {} features {}.'.format(
    f_classif_selected.sum(), f_classif_selected_features))

selected = chi2_selected & f_classif_selected
print('Chi2 & F_classif selected {} features'.format(selected.sum()))

features = [f for f, s in zip(X.columns, selected) if s]
print(features)
X_sel = X[features]

# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_sel,
#   y, random_state=1301, stratify=y, test_size=0.33)
   
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=800, random_state=1301, n_jobs=-1,
   criterion='gini', class_weight='balanced', max_depth=10)

scores = defaultdict(list)

X = X.as_matrix()
y = y.as_matrix()
X_sel = X_sel.as_matrix()

# Based on http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
#crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=1301):
    X_train, X_test = X_sel[train_idx], X_sel[test_idx]
    Y_train, Y_test = y[train_idx], y[test_idx]
    r = rfc.fit(X_train, Y_train)
    auc = roc_auc_score(Y_test, rfc.predict_proba(X_test)[:,1])
    for i in range(X_sel.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_auc = roc_auc_score(Y_test, rfc.predict_proba(X_t)[:,1])
        scores[features[i]].append((auc-shuff_auc)/auc)
print ("Features sorted by their score:")
print (sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))

rfc_features = [feat for feat in scores.keys()]

print (rfc_features)

ts = pd.DataFrame({'feature': rfc_features,
                   'score': [np.mean(score) for score in scores.values()],
                   })

featp = ts.sort_values(by='score')[-20:].plot(kind='barh', x='feature', y='score', legend=False, figsize=(6, 10))
plt.title('Random Forest Classifier Feature Importance')
fig_featp = featp.get_figure()
fig_featp.savefig('feature_importance_rfc.png', bbox_inches='tight', pad_inches=1)
test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]
rfc.fit(X_sel, y)
y_pred = rfc.predict_proba(sel_test)

submission = pd.DataFrame({"ID":test.index, "TARGET":y_pred[:,1]})
submission.to_csv("submission_rfc.csv", index=False)
