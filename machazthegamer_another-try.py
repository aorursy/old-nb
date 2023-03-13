#!/usr/bin/env python
# coding: utf-8



"""
for both:
took the data cleaning code from the forums

for split and joint:
fill nans with 99
add some features
fill the infs and nans of new features
split train data by product type and train different models for them and get predictions
compare the relative importance of the features in each of the models
form new interaction features with product type and train a joint model then use this to fill for 
predictions with missing product type.

for interact and randomize:
Because previous performed poorly, I do a randomizedlasso on combined df with the interaction features
then try to predict based on the transformed one

for split interact and average:
I average the predictions for the previous two because they both get low LB scores

"""




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.




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




#look at some stats for the different product types
import scipy.stats as stats
prod = combined[["product_type"]][:30471]
prod["price_doc"]=df.price_doc
prod.pivot_table(index=["product_type"], values=["price_doc"], aggfunc=np.median)




# add month, monthyear, ratio of green to industry, ofice count over leisure, 
#muslim to christ, ratio of close to far big chuches, mean distance to the art attractions
#increase in buildings from old_times, ratio of children to youngens, and ratio of young male to female 
combined["timestamp"] = pd.to_datetime(combined.timestamp)
combined = combined.assign(month=combined.timestamp.dt.month)
combined = combined.assign(month_year=combined.timestamp.dt.year.astype(str) + combined.month.astype(str))           .assign(green_industry=combined["green_part_3000"]/combined["prom_part_3000"])           .assign(work_or_play=combined["office_count_1500"]/(combined["sport_count_1500"]+combined["leisure_count_1500"]))           .assign(islam_or_christ=combined["mosque_count_500"]/ combined["church_count_500"])           .assign(church_appeal=combined["big_church_count_500"]/combined["big_church_count_1500"])           .assign(mean_km_art=combined[["museum_km", "exhibition_km", "catering_km", "theater_km", "park_km"]].mean(axis=1))           .assign(new_to_old_count=combined["build_count_after_1995"]/combined["build_count_1971-1995"])           .assign(new_to_older_count=combined["build_count_after_1995"]/combined["build_count_1946-1970"])        .assign(young_to_old=combined["0_13_all"]/combined["16_29_all"])        .assign(male_to_femal_young=combined["young_male"]/combined["young_female"])
        
combined["month_year"] = combined.month_year.astype(int)




counts = combined.groupby("month_year")["id"].count()
df4 = pd.DataFrame(counts).reset_index()
df4.columns = ["month_year", "sales_count"]
combined = combined.merge(df4, on="month_year")




for i, col in enumerate(combined.columns.tolist()):
    try:
        if col != "timestamp" or col != "month_year" or col != "month":
            combined.iloc[np.where(combined[col].isnull().values)[0], i]=0
            combined.iloc[np.where(np.isinf(combined[col].values))[0], i] = 99
    except TypeError:
        continue





no_type = combined.loc[combined["product_type"]==99, "id"].values #all are from test
train_0_id = combined[:30471].query("product_type == 0")["id"].values
train_1_id = combined[:30471].query("product_type == 1")["id"].values
test_0_id = combined[30471:].query("product_type == 0")["id"].values
test_1_id = combined[30471:].query("product_type == 1")["id"].values
test_all_id = combined[30471:].id.values




#split train test samples for each product type
combined = combined.drop(labels=["id", "timestamp"], axis=1, errors="ignore")
x_train_all = combined[:30471].copy()
x_test_all = combined[30471:]
x_train_all.loc[:, "price_doc"] =  df.price_doc.values

x_train_0 = x_train_all.query("product_type == 0")
price_0 = x_train_0.price_doc.values

x_train_1 = x_train_all.query("product_type == 1")
price_1 = x_train_1.price_doc.values

x_test_0 = x_test_all.query("product_type == 0")
x_test_1 = x_test_all.query("product_type == 1")




from sklearn.ensemble import GradientBoostingRegressor
reg_0 = GradientBoostingRegressor(max_depth=2, min_samples_split=10, min_samples_leaf=8)
reg_1 = GradientBoostingRegressor(max_depth=2, min_samples_split=10, min_samples_leaf=8)

reg_0.fit(X=x_train_0.drop(labels=["price_doc"], axis=1).values, y=price_0)
reg_1.fit(X=x_train_1.drop(labels=["price_doc"], axis=1).values, y=price_1)

#ridge regression for whole sample
from sklearn.linear_model import RidgeCV
reg_all = RidgeCV(alphas=(0.6,0.7,0.8,0.9,1))
reg_all.fit(X=x_train_all.drop(labels=["price_doc"], axis=1).values, y=df.price_doc.values)


#predictions
predictions_0 = reg_0.predict(X=x_test_0)
predictions_1 = reg_1.predict(X=x_test_1)
predictions_all = reg_all.predict(X=x_test_all.values)

#put into Series
s0 = pd.Series(data=predictions_0, index=test_0_id, name="price_doc")
s1 = pd.Series(data=predictions_1, index=test_1_id, name="price_doc")
pred_01 = s0.append(s1)


s3 = pd.Series(data=predictions_all, index=test_all_id, name="price_doc")
s4 = s3.loc[no_type]
pred_all = pred_01.append(s4)
print(pred_all.head(5))
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
tr_pred_0 = reg_0.predict(X=x_train_0.drop(labels=["price_doc"], axis=1).values)
tr_pred_1 = reg_1.predict(x_train_1.drop(labels=["price_doc"], axis=1).values)
plt.hist([tr_pred_0, tr_pred_1], bins=60, label=["0_type", "1_type"], histtype="stepfilled", alpha=0.5)
plt.title("Histogram of 1 and 0 type predictions")
plt.legend()




tr_pred_all = reg_all.predict(X=x_train_all.drop(labels=["price_doc"], axis=1).values)
dic = {"pred_price": np.insert(arr=tr_pred_1, values=tr_pred_0, obj=0), 
       "id":np.insert(arr=train_1_id, values=train_0_id, obj=0), 
       "month_year":np.insert(arr=x_train_1["month_year"].values, values=x_train_0["month_year"].values, obj=0)}
df1 = pd.DataFrame(data=dic)
dic2 = {"pred_price_all": tr_pred_all, "id": df.id.values, "month_year":x_train_all["month_year"].values}
df2 = pd.DataFrame(data=dic2)




mean1 = df1.groupby(by="month_year").mean().sort_index()
mean2 = df2.groupby(by="month_year").mean().sort_index()
mean_actual = x_train_all.groupby("month_year")["price_doc"].mean().sort_index()
fig, ax = plt.subplots(1)
ax.plot(range(mean1.index.shape[0]), mean1.pred_price.values, "-", 
       range(mean1.index.shape[0]), mean_actual.values, "--")
ax.legend(["split_predictions", "actual"])




print(reg_0.score(X=x_train_0.drop(labels=["price_doc"], axis=1).values,y=x_train_0.price_doc.values))
reg_1.score(X=x_train_1.drop(labels=["price_doc"], axis=1).values,y=x_train_1.price_doc.values)




f_imp = pd.DataFrame(index=x_train_0.drop(labels=["price_doc"], axis=1).columns.values, 
          data={"imp_0":reg_0.feature_importances_, 
               "imp_1":reg_1.feature_importances_})




plt.rcParams["figure.figsize"]=30,10
f_imp[f_imp["imp_1"]>0].sort_values("imp_1").plot(kind="bar")


    
    
#unimportant = f_imp[f_imp["imp_1"]==0].index[:100]
#combined2 = combined[[x for x in combined.columns if x not in unimportant]]




#some additional_vars from the difference in importance
#this will be interactions btn prod type and features
interact = ["full_sq", "build_year", "life_sq", "floor", "thermal_power_plant_km", "ttk_km", 
          "mean_km_art", "metro_min_avto", "university_km", "prom_part_2000"]
pt = combined.product_type.values
for name in interact:
    combined[name+"*pt"] = combined[name] * pt
    

#different direction starts here
train_interact = combined[:30471]
test_interact = combined[30471:]
log_price = np.log(df.price_doc.values)




from sklearn.linear_model import RandomizedLasso
rl = RandomizedLasso()
rl.fit(X=train_interact.values, y=log_price)


rl_train = rl.transform(train_interact.values)
rl_test = rl.transform(test_interact.values)




from sklearn.ensemble import GradientBoostingRegressor
reg_interact = GradientBoostingRegressor(max_depth=2, min_samples_split=10, min_samples_leaf=8)
reg_interact.fit(X=rl_train, y=log_price)
pred_interact = reg_interact.predict(X=rl_test)
pred_series = pd.Series(data=np.exp(pred_interact), index=test.id.values, name="price_doc")




reg_interact.score(X=rl_train, y=log_price)




pred_series.head()




pred_all = pred_all.sort_index()
averaged = (pred_all+pred_series)/2




averaged.to_csv("interact and randomize.csv", header=True, index_label="id")




"The end"






