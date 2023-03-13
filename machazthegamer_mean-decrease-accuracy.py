#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np




df = pd.read_csv("../input/train.csv")
df_x = df.drop(labels=["price_doc"], axis=1)
test = pd.read_csv("../input/test.csv")

train = df_x
#data cleaning
bad_index = train[train.life_sq > train.full_sq].index
train.ix[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test.ix[equal_index, "life_sq"] = test.ix[equal_index, "full_sq"]
bad_index = test[test.life_sq > test.full_sq].index
test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train[train.life_sq < 5].index
train.ix[bad_index, "life_sq"] = np.NaN
bad_index = test[test.life_sq < 5].index
test.ix[bad_index, "life_sq"] = np.NaN
bad_index = train[train.full_sq < 5].index
train.ix[bad_index, "full_sq"] = np.NaN
bad_index = test[test.full_sq < 5].index
test.ix[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train.ix[kitch_is_build_year, "build_year"] = train.ix[kitch_is_build_year, "kitch_sq"]
bad_index = train[train.kitch_sq >= train.life_sq].index
train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = test[test.kitch_sq >= test.life_sq].index
test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.kitch_sq == 0).values + (train.kitch_sq == 1).values].index
train.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = test[(test.kitch_sq == 0).values + (test.kitch_sq == 1).values].index
test.ix[bad_index, "kitch_sq"] = np.NaN
bad_index = train[(train.full_sq > 210) & (train.life_sq / train.full_sq < 0.3)].index
train.ix[bad_index, "full_sq"] = np.NaN
bad_index = test[(test.full_sq > 150) & (test.life_sq / test.full_sq < 0.3)].index
test.ix[bad_index, "full_sq"] = np.NaN
bad_index = train[train.life_sq > 300].index
train.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
bad_index = test[test.life_sq > 200].index
test.ix[bad_index, ["life_sq", "full_sq"]] = np.NaN
train.product_type.value_counts(normalize= True)
test.product_type.value_counts(normalize= True)
bad_index = train[train.build_year < 1500].index
train.ix[bad_index, "build_year"] = np.NaN
bad_index = test[test.build_year < 1500].index
test.ix[bad_index, "build_year"] = np.NaN
bad_index = train[train.num_room == 0].index 
train.ix[bad_index, "num_room"] = np.NaN
bad_index = test[test.num_room == 0].index 
test.ix[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train.ix[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test.ix[bad_index, "num_room"] = np.NaN
bad_index = train[(train.floor == 0).values * (train.max_floor == 0).values].index
train.ix[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train[train.floor == 0].index
train.ix[bad_index, "floor"] = np.NaN
bad_index = train[train.max_floor == 0].index
train.ix[bad_index, "max_floor"] = np.NaN
bad_index = test[test.max_floor == 0].index
test.ix[bad_index, "max_floor"] = np.NaN
bad_index = train[train.floor > train.max_floor].index
train.ix[bad_index, "max_floor"] = np.NaN
bad_index = test[test.floor > test.max_floor].index
test.ix[bad_index, "max_floor"] = np.NaN
train.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train.ix[bad_index, "floor"] = np.NaN
train.material.value_counts()
test.material.value_counts()
train.state.value_counts()
bad_index = train[train.state == 33].index
train.ix[bad_index, "state"] = np.NaN
test.state.value_counts()


df_x = train 
combined = pd.concat([df_x,test], ignore_index=True, axis=0)
obj_col = combined.select_dtypes(include=[object]).columns

from sklearn.preprocessing import LabelEncoder
for name in obj_col:
    if name != "timestamp" and name != "product_type":
        print(name)
        encoder = LabelEncoder()
        combined[name] = encoder.fit_transform(combined[name].fillna(value=99).values)
        
#investment is 1 while occupier 0        
combined["product_type"] = combined.product_type.map({"Investment":1, 
                                                      "OwnerOccupier":0, np.nan:99}).values

filled= combined.groupby(by="sub_area").fillna(99)
combined = filled.merge(combined[["id", "sub_area"]], on="id")

#add ratio and age vars
combined["age"] = pd.to_datetime(combined["timestamp"]).dt.year - combined.build_year
combined["ratio"] = (combined["full_sq"]/combined["life_sq"]).fillna(value=99)
combined = combined.replace([-np.inf, np.inf], 99)





# add month, monthyear, ratio of green to industry, ofice count over leisure, 
#muslim to christ, ratio of close to far big chuches, mean distance to the art attractions
#increase in buildings from old_times, ratio of children to youngens, and ratio of young male to female 
combined["timestamp"] = pd.to_datetime(combined.timestamp)
combined = combined.assign(month=combined.timestamp.dt.month)
combined = combined.assign(month_year=combined.timestamp.dt.year.astype(str) + combined.month.astype(str))           .assign(green_industry=combined["green_part_3000"]/combined["prom_part_3000"])           .assign(work_or_play=combined["office_count_1500"]/(combined["sport_count_1500"]+combined["leisure_count_1500"]))           .assign(islam_or_christ=combined["mosque_count_500"]/ combined["church_count_500"])           .assign(church_appeal=combined["big_church_count_500"]/combined["big_church_count_1500"])           .assign(mean_km_art=combined[["museum_km", "exhibition_km", "catering_km", "theater_km", "park_km"]].mean(axis=1))           .assign(new_to_old_count=combined["build_count_after_1995"]/combined["build_count_1971-1995"])           .assign(new_to_older_count=combined["build_count_after_1995"]/combined["build_count_1946-1970"])        .assign(young_to_old=combined["0_13_all"]/combined["16_29_all"])        .assign(male_to_femal_young=combined["young_male"]/combined["young_female"])
        
combined["month_year"] = combined.month_year.astype(np.float64)
combined = combined.replace([-np.inf, np.inf], 99)




combined = combined.drop(labels=["id", "timestamp"], axis=1, errors="ignore").fillna(value=1)
x_traindf = combined[:30471].copy()
x_testdf = combined[30471:]
price =  df.price_doc.values




from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_traindf.values, price, train_size=0.7, test_size=0.3)




from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge, PassiveAggressiveRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from collections import defaultdict

regressors = {"linreg" : LinearRegression(), 
              "sdg" : SGDRegressor(l1_ratio=0, alpha=0.01), 
              "bayes" : BayesianRidge(), 
              "pasagg" : PassiveAggressiveRegressor(), 
              "neural": MLPRegressor(), 
             "gtr":GradientBoostingRegressor()}


columns = x_traindf.columns
 




splitter = ShuffleSplit(test_size=0.2, n_splits=10)
def mean_accuracy(estimator, train, target, reg=None):
    """
    train and targets should be numpy arrays
    """
    scores = []
    decreases = defaultdict(list)
    for train_index, test_index in splitter.split(X=train, y=target):
        train_x, train_y = train[train_index, :], target[train_index]
        test_x, test_y = train[test_index, :], target[test_index]
    
        estimator.fit(X=train_x, y=train_y)
        score = estimator.score(X=test_x, y=test_y)
        scores.append(score)
        for i, name in enumerate(columns):
            test_copy = np.copy(test_x)
            np.random.shuffle(test_copy[:, i])
            try:
                test_copy = estimator.transform(test_copy)
            except AttributeError:
                pass
            score_i = estimator.score(X=test_copy, y=price[test_index])
            decrease = (score_i - score)/score
            decreases[name].append(decrease)
    print(reg)
    print(np.median(scores))
    return pd.DataFrame(data=decreases).mean()




gtb_imp = mean_accuracy(regressors["gtr"], X_train, Y_train, reg="gtr").nlargest(100).index.to_list()




#important = {}
#for k in regressors.keys():
#    important[k] = mean_accuracy(regressors[k], X_train, Y_train, reg=k).nlargest(100).index.to_list()




from sklearn.decomposition import KernelPCA
decomposer = KernelPCA(n_components=20, kernel="poly", copy_X=False)
col_indices = pd.Series(data=range(len(columns)), index=columns)
#performance = {}
def validate(reg, col_imp):
    """giv fit regressor and its important columns"""
    col_index = col_indices.loc[col_imp].values
    for_pca = set(columns).difference(col_imp)
    pca_index = col_indices.loc[for_pca].values
    X_train_copy = X_train.copy()
    X_test_copy = X_test.copy()
    decomposer.fit_transform(X_train_copy[:, pca_index])
    X_test_copy[:, ] = decomposer.transform(X_test_copy[:, pca_index])
    reg.fit(X=X_train_copy, y=Y_train)
    score = regressors[k].score(X=X_test_copy, y=Y_test)  
    return score




validate(regressors["gtr"], gtb_imp)






