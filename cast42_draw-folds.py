import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score

training = pd.read_csv("../input/train.csv", index_col=0)
test = pd.read_csv("../input/test.csv", index_col=0)

print(training.shape)
print(test.shape)

# Replace -999999 in var3 column with most common value 2 
# See https://www.kaggle.com/cast42/santander-customer-satisfaction/debugging-var3-999999
# for details
training = training.replace(-999999,2)

X = training.iloc[:,:-1]
y = training.TARGET

# Add zeros per row as extra feature
X['n0'] = (X == 0).sum(axis=1)

# Select the features calculated in https://www.kaggle.com/cast42/santander-customer-satisfaction/select-features-rfecv/code
# 
# features = \
# ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 
# 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 
# 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var8_0', 'ind_var30', 'num_var4',
# 'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_ult1', 
# 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var37', 
# 'num_var39_0', 'num_var42', 'saldo_var5', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 
# 'imp_trans_var37_ult1', 'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 
# 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 
# 'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_comer_ult1',
# 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult3', 
# 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3',
# 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'var38', 'n0']

# features = ['var15', 'saldo_var30', 'saldo_var42', 'num_var22_hace2', 'num_var22_ult3',
# 'num_meses_var5_ult3', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3',\
# 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 
# 'var38', 'n0']

features = ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult3', 
'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 
'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_ult1', 'ind_var8_0', 
'ind_var30_0', 'ind_var30', 'num_op_var41_hace2', 'num_op_var41_ult3', 
'num_var37_med_ult2', 'saldo_var5', 'saldo_var8', 'saldo_var26', 'saldo_var30', 
'saldo_var37', 'saldo_var42', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 
'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3',
'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3',
'num_op_var39_comer_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1',
'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1',
'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 
'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2',
'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_ult3',
'saldo_medio_var13_corto_hace2', 'var38', 'n0']

X_sel = X[features]



nrfold = 2

skf = cross_validation.StratifiedKFold(y, n_folds=nrfold, shuffle=True, random_state=1301)

scores = []
X = X.as_matrix()
y = y.as_matrix()
X_sel = X_sel.as_matrix()
test['n0'] = (test == 0).sum(axis=1)
sel_test = test[features]

y_pred_list = []
y_train_pred_list = []
for k, (train,valid) in enumerate(skf):
    print ('Train on fold {}'.format(k+1))
    ratio = float(np.sum(y == 1)) / np.sum(y==0)
    clf = xgb.XGBClassifier(missing=9999999999,
                    max_depth = 7,
                    n_estimators=500,
                    learning_rate=0.1, 
                    nthread=4,
                    subsample=1.0,
                    colsample_bytree=0.5,
                    min_child_weight = 3,
                    scale_pos_weight = ratio,
                    seed=1301)
    clf.fit(X_sel[train], y[train], early_stopping_rounds=50, eval_metric="auc",
        eval_set=[(X_sel[train], y[train]), (X_sel[valid], y[valid])])
    y_train_pred = clf.predict_proba(X_sel, ntree_limit=clf.best_iteration)
    y_train_pred_list.append(y_train_pred)
    score = roc_auc_score( y[valid], y_valid_pred[:,1])
    scores.append((score, clf.best_iteration))
    print('Fold: %d, Dist. %s, #Rounds: %d AUC: %.3f' % (k+1, np.bincount(y[train]), clf.best_iteration, score))
    print('Predicting test values')
    y_pred = clf.predict_proba(sel_test, ntree_limit=clf.best_iteration)
    y_pred_list.append(y_pred)

auc_scores =  pd.Series([score[0] for score in scores])
print ('Avg valid AUC sore %.4f +/- %.4f' % (auc_scores.mean(), auc_scores.std()))

pred = np.mean(np.array(y_pred_list), axis = 0)

np.array(y_pred_list).shape

np.mean(np.array(y_pred_list)[:,:,1], axis = 0).shape

test['std'] = np.std(np.array(y_pred_list)[:,:,1], axis = 0)

df = pd.DataFrame(test[['std']].sort_values(by='std', ascending=False))

df.plot(kind='bar')

len(features)

X = np.array(y_pred_list)[:,:,1] 

X.shape

X[0,:].shape

Xt = np.transpose(X)

from sklearn.decomposition import PCA
pca = PCA(n_components= 2)
X_pca  = pca.fit_transform (sel_test)
plt.figure()

# http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
   '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
for k in range(X.shape[0]):
    plt.scatter(X_pca[:,0], X_pca[:,1], c = X[k,:], cmap = 'hot')


plt.savefig('fold_position.png', bbox_inches='tight', pad_inches=1)

# What custormers make AUC drop
plt.figure()
X_pca  = pca.fit_transform (X_sel)
# http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
   '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
for k, (train,valid) in enumerate(skf):
    if k in [10, 11]:
        plt.scatter(X_pca[valid,0], X_pca[valid,1], c = 'r', s=50)
    else:
        plt.scatter(X_pca[valid,0], X_pca[valid,1], c = 'b', alpha=0.2, linewidth = 0)

plt.savefig('fold_position.png', bbox_inches='tight', pad_inches=1)

X_pca.shape

pca.components_[0]

len(pca.components_[0])

plt.figure(figsize=(8, 14))
_ = plt.barh(np.arange(len(pca.components_[0])), -pca.components_[0])
pos = np.arange(len(pca.components_[0]))
ticks = plt.yticks(pos + .5, features[:len(pca.components_[0])])
for p, c, ch in zip(pos, features, -pca.components_[0]):
    if ch > 0:
        plt.annotate("%.2f" % ch, xy=(ch +0.005, p + .5), va='center')
    else:
        plt.annotate("%.2f" % ch, xy=(ch - 0.04, p + .5), va='center')
# plt.xlim(-0.14,0.35)
# plt.ylim(0,len(pca.components_[0])-1)
plt.show()

import scipy.stats as st

gpred = st.gmean(np.array(y_valid_pred_list), axis = 0)

gpred

np.array(y_valid_pred_list)[2][:,1]

np.array(y_valid_pred_list).shape

# for k, (train,valid) in enumerate(skf):

np.mean(np.array(y_train_pred_list), axis=0)[:,1]

X['mean'] =np.mean(np.array(y_train_pred_list), axis=0)[:,1]

train['std']


