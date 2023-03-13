#!/usr/bin/env python
# coding: utf-8



import gc
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Inputs
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Data viz
from mlens.visualization import corr_X_y, corrmat

# Model evaluation
from mlens.metrics import make_scorer
from mlens.model_selection import Evaluator

# Ensemble
from mlens.ensemble import SuperLearner

from scipy.stats import uniform, randint

from matplotlib.pyplot import show
get_ipython().run_line_magic('matplotlib', 'inline')




SEED = 148
np.random.seed(SEED)




def build_train():
    """Read in training data and return input, output, columns tuple."""

    # This is a version of Anovas minimally prepared dataset
    # for the xgbstarter script
    # https://www.kaggle.com/anokas/simple-xgboost-starter-0-0655

    df = pd.read_csv('../input/train_2016_v2.csv')

    prop = pd.read_csv('../input/properties_2016.csv')
    convert = prop.dtypes == 'float64'
    prop.loc[:, convert] =         prop.loc[:, convert].apply(lambda x: x.astype(np.float32))

    df = df.merge(prop, how='left', on='parcelid')

    y = df.logerror
    df = df.drop(['parcelid',
                  'logerror',
                  'transactiondate',
                  'propertyzoningdesc',
                  'taxdelinquencyflag',
                  'propertycountylandusecode'], axis=1)

    convert = df.dtypes == 'object'
    df.loc[:, convert] =         df.loc[:, convert].apply(lambda x: 1 * (x == True))

    df.fillna(0, inplace=True)

    return df, y, df.columns




xtrain, ytrain, columns = build_train()
xtrain, xtest, ytrain, ytest = train_test_split(
    xtrain, ytrain, test_size=0.5, random_state=SEED)




corr_X_y(xtrain, ytrain, figsize=(16, 10), label_rotation=80, hspace=1, fontsize=14)




# We consider the following models (or base learners)
gb = XGBRegressor(n_jobs=1, random_state=SEED)
ls = Lasso(alpha=1e-6, normalize=True)
el = ElasticNet(alpha=1e-6, normalize=True)
rf = RandomForestRegressor(random_state=SEED)

base_learners = [
    ('ls', ls), ('el', el), ('rf', rf), ('gb', gb)
]




P = np.zeros((xtest.shape[0], len(base_learners)))
P = pd.DataFrame(P, columns=[e for e, _ in base_learners])

for est_name, est in base_learners:
    est.fit(xtrain, ytrain)
    p = est.predict(xtest)
    P.loc[:, est_name] = p
    print("%3s : %.4f" % (est_name, mean_absolute_error(ytest, p)))




ax = corrmat(P.corr())
show()




# Put their parameter dictionaries in a dictionary with the
# estimator names as keys
param_dicts = {
    'ls':
    {'alpha': uniform(1e-6, 1e-5)},
    'el':
    {'alpha': uniform(1e-6, 1e-5),
     'l1_ratio': uniform(0, 1)
    },
    'gb':
    {'learning_rate': uniform(0.02, 0.04),
     'colsample_bytree': uniform(0.55, 0.66),
     'min_child_weight': randint(30, 60),
     'max_depth': randint(3, 7),
     'subsample': uniform(0.4, 0.2),
     'n_estimators': randint(150, 200),
     'colsample_bytree': uniform(0.6, 0.4),
     'reg_lambda': uniform(1, 2),
     'reg_alpha': uniform(1, 2),
    },
    'rf':
    {'max_depth': randint(2, 5),
     'min_samples_split': randint(5, 20),
     'min_samples_leaf': randint(10, 20),
     'n_estimators': randint(50, 100),
     'max_features': uniform(0.6, 0.3)
    }
}




scorer = make_scorer(mean_absolute_error, greater_is_better=False)

evl = Evaluator(
    scorer,
    cv=2,
    random_state=SEED,
    verbose=5,
)




evl.fit(
    xtrain, ytrain,
    estimators=base_learners,
    param_dicts=param_dicts,
    preprocessing={'sc': [StandardScaler()], 'none': []},
    n_iter=2  # bump this up to do a larger grid search
)




pd.DataFrame(evl.results)




evl.results["params"]['sc.gb']




for case_name, params in evl.results["params"].items():
    case, case_est = case_name.split('.')
    for est_name, est in base_learners:
        if est_name == case_est:
            est.set_params(**params)




# We will compare a GBM and an elastic net as the meta learner
# These are cloned internally so we can go ahead and grab the fitted ones
meta_learners = [
    ('gb', gb), ('el', el)
]

# Note that when we have a preprocessing pipeline,
# keys are in the (prep_name, est_name) format
param_dicts = {
    'el':
    {'alpha': uniform(1e-5, 1),
     'l1_ratio': uniform(0, 1)
    },
    'gb':
    {'learning_rate': uniform(0.01, 0.2),
     'subsample': uniform(0.5, 0.5),
     'reg_lambda': uniform(0.1, 1),
     'n_estimators': randint(10, 100)
    },
}




# Put the layers you don't want to tune into an ensemble with model selection turned on
# Just remember to turn it off when you're done!
in_layer = SuperLearner(model_selection=True)
in_layer.add(base_learners)

preprocess = [in_layer]




evl.fit(
    xtrain, ytrain,
    meta_learners,
    param_dicts,
    preprocessing={'meta': preprocess},
    n_iter=4                            # bump this up to do a larger grid search
)




pd.DataFrame(evl.results)




# Let's pick the linear meta learner with the above tuned
# hyper-parameters. Note that ideally, you'd want to tune
# the ensemble as a whole, not each estimator at a time
meta_learner = meta_learners[1][1]
meta_learner.set_params(**evl.results["params"]["meta.el"])

# We can grab the preprocessing layer and turn model selection off
ens = in_layer
ens.model_selection = False
ens.add_meta(meta_learner)




ens.fit(xtrain, ytrain)




pred = ens.predict(xtest)




print("ensemble score: %.4f" % mean_absolute_error(ytest, pred))

