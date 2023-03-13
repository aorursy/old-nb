#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
ml_ch=pd.read_csv("../input/mulliken_charges.csv")
dp_mts=pd.read_csv("../input/dipole_moments.csv")
structures=pd.read_csv("../input/structures.csv")
mngt_she_tensors=pd.read_csv("../input/magnetic_shielding_tensors.csv")
pot_ener=pd.read_csv("../input/potential_energy.csv")
scc=pd.read_csv("../input/scalar_coupling_contributions.csv")




print(train_data.shape)
print(train_data.molecule_name.nunique())
train_data.head()




train_data[train_data['molecule_name']=='dsgdb9nsd_000001']




train_data.atom_index_0.value_counts()




mols_train=train_data['molecule_name'].unique()




print(dp_mts.shape)
print(dp_mts.molecule_name.nunique())
dp_mts.head()









print(ml_ch.shape)
print(ml_ch.molecule_name.nunique())
ml_ch.head()




print(structures.shape)
print(structures.molecule_name.nunique())
structures.head()




structures[structures['molecule_name']=='dsgdb9nsd_000001']




print(test_data.shape)
print(test_data.molecule_name.nunique())
test_data.head()




test_data.atom_index_0.value_counts()




test_data[test_data['molecule_name']=='dsgdb9nsd_000004']








mols_test=test_data['molecule_name'].unique()




(mols_train[:45772]) & (mols_test)




mols_test









print(mngt_she_tensors.shape)
print(mngt_she_tensors.molecule_name.nunique())
mngt_she_tensors.head(10)




mngt_she_tensors[mngt_she_tensors['molecule_name']=='dsgdb9nsd_000001']









print(pot_ener.shape)
print(pot_ener.molecule_name.nunique())
pot_ener.head()





train_data['potential_energy']=pot_ener['molecule_name']==i









print(scc.shape)
print(scc.molecule_name.nunique())
scc.head()




scc[scc['molecule_name']=='dsgdb9nsd_000001']




structures.groupby(['molecule_name','atom'])['atom'].size()



train_data.dtypes




train_data['atom_index_0']=train_data['atom_index_0'].astype('category')
train_data['atom_index_1']=train_data['atom_index_1'].astype('category')




train_data.dtypes




train_data[train_data['atom_index_0']==0]
train_data.drop([1600734,1600735,1600736],inplace=True)




dummies_type_train=pd.get_dummies(train_data['type'])
dummies_index0_train=pd.get_dummies(train_data['atom_index_0'],prefix='0_index')
dummies_index1_train=pd.get_dummies(train_data['atom_index_1'],prefix='1_index')




print(dummies_index0_train.shape)
print(dummies_index1_train.shape)




dummies_index0_train.columns




dummies_type_test=pd.get_dummies(test_data['type'])
dummies_index0_test=pd.get_dummies(test_data['atom_index_0'],prefix='0_index')
dummies_index1_test=pd.get_dummies(test_data['atom_index_1'],prefix='1_index')




dummies_index0_test.columns




print(dummies_index0_test.shape)
print(dummies_index1_test.shape)




train_data[train_data.atom_index_0=='0']




dummies_index0_train=dummies_index0_train.drop('0_index_0',axis=1)




X_train=pd.concat([dummies_type_train,dummies_index0_train,dummies_index1_train],axis=1).values




X_train.shape




X_test=pd.concat([dummies_type_test,dummies_index0_test,dummies_index1_test],axis=1).values




y_train=train_data['scalar_coupling_constant'].values




y_train.shape




from sklearn.linear_model import Ridge




ridge = Ridge(alpha=0.1, normalize=True) 




ridge.fit(X_train,y_train)




y_pred=ridge.predict(X_test)




sample=pd.read_csv("../input/sample_submission.csv")




sample.columns




sample['scalar_coupling_constant']=y_pred




sample.to_csv('sample.csv',index=False)






