from sklearn import metrics, cross_validation, linear_model
from scipy import sparse
from itertools import combinations
import numpy as np
import pandas as pd
SEED = 123

def group_data(data, degree=3, hash=hash):
    new_data = []
    m, n = data.shape
    for indices in combinations(range(n), degree):
        new_data.append([hash(tuple(v)) for v in data[:, indices]])
    return np.array(new_data).T


def OneHotEncoder(data, keymap=None):
    """
    OneHotEncoder takes data matrix with categorical columns and
    converts it to a sparse binary matrix.

    Returns sparse binary matrix and keymap mapping categories to indicies.
    If a keymap is supplied on input it will be used instead of creating one
    and any categories appearing in the data that are not in the keymap are
    ignored
    """
    if keymap is None:
        keymap = []
        for col in data.T:
            uniques = set(list(col))
            keymap.append(dict((key, i) for i, key in enumerate(uniques)))
    total_pts = data.shape[0]
    outdat = []
    for i, col in enumerate(data.T):
        km = keymap[i]
        num_labels = len(km)
        spmat = sparse.lil_matrix((total_pts, num_labels))
        for j, val in enumerate(col):
            if val in km:
                spmat[j, km[val]] = 1
        outdat.append(spmat)
    outdat = sparse.hstack(outdat).tocsr()
    return outdat, keymap


def cv_loop(X, y, model, N):
    mean_auc = 0.
    for i in range(N):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            X, y, test_size=.20,
            random_state=i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_cv)[:, 1]
        auc = metrics.auc_score(y_cv, preds)
        #print "AUC (fold %d/%d): %f" % (i + 1, N, auc)
        mean_auc += auc
    return mean_auc/N



dp = group_data(all_data, degree=2)
train_data.shape



dt = group_data(all_data, degree=3)

y = np.array(train_data['Response'].values)


print "anything"

X = all_data[:num_train]
X_2 = dp[:num_train]
X_3 = dt[:num_train]

X_3.shape

num_features = X_2.shape[1]
print(num_features)
