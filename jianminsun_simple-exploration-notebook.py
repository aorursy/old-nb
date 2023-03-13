#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)




data_path = "../input/"
train = pd.read_csv(data_path+"train.csv", usecols=['ncodpers'])
test = pd.read_csv(data_path+"test.csv", usecols=['ncodpers'])
print("Number of rows in train : ", train.shape[0])
print("Number of rows in test : ", test.shape[0])




train_unique_customers = set(train.ncodpers.unique())
test_unique_customers = set(test.ncodpers.unique())
print("Number of customers in train : ", len(train_unique_customers))
print("Number of customers in test : ", len(test_unique_customers))
print("Number of common customers : ", len(train_unique_customers.intersection(test_unique_customers)))




sub = pd.read_csv(data_path+"sample_submission.csv")
print("Number of rows in sample submission file : ",sub.shape[0])
sub_unique_customers = set(sub.ncodpers.unique())
print("Number of unique customers in sample submission file : ",len(sub_unique_customers))
print("Number of customers from test missing in sample sub :", len(test_unique_customers - sub_unique_customers))




del train_unique_customers
del test_unique_customers
del sub
del sub_unique_customers




train = pd.read_csv(data_path+"train.csv",  usecols=['ncodpers', 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1'])




train['ncodpers'][:10]




data_path = "../input/"
train = pd.read_csv(data_path+"train.csv", usecols=['ncodpers'], nrows=1000)




train = pd.read_csv(data_path+"train.csv",  usecols=['ncodpers', 'ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1'])




#train = pd.read_csv(data_path+"train.csv",  usecols=["fecha_dato", 'ncodpers'])
df = train[train['ncodpers']==1050615].astype(int)




test = pd.read_csv(data_path+"test.csv",  usecols=['ncodpers'])




test[test['ncodpers']==1050612]




df.set_index('ncodpers').diff()






