#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




dftrain = pd.read_csv('train_2016.csv')




dfprop = pd.read_csv('properties_2016.csv')




dfprop.describe()




dfprop.index = dfprop.parcelid




dfprop = dfprop.drop('parcelid',axis=1)




dfprop.head()




dftrain.index = dftrain.parcelid




dftrain = dftrain.drop('parcelid',axis=1)




dftrain.head()




dfcomb = dfprop.join(dftrain,how='outer')




dfcomb.head()




np.sum(np.isnan(dfcomb.logerror))




dfcomb_nar = dfcomb.loc[pd.isnull(dfcomb.logerror) == False,:]




dfcomb_nar.shape




desc_df = pd.DataFrame(index = ['% nan','num unique vals'],columns = dfcomb_nar.columns)
for col in dfcomb_nar.columns:
    desc_df.loc['% nan',col] = np.sum(pd.isnull(dfcomb_nar.loc[:,col]))/dfcomb_nar.shape[0]
    desc_df.loc['num unique vals',col] = len(dfcomb_nar.loc[:,col].value_counts())
    
    




desc_df.iloc[:,:10]




desc_df.iloc[:,10:20]




desc_df.iloc[:,20:30]




desc_df.iloc[:,30:40]




desc_df.iloc[:,40:50]




desc_df.iloc[:,50:60]




dfprop['parcelid'].head()




dfprop.propertycountylandusecode.value_counts()




dfprop['airconditioningtypeid'].head()




dfprop['airconditioningtypeid'].value_counts()




dfprop['architecturalstyletypeid'].head()




dfprop['architecturalstyletypeid'].tail()




dfprop['architecturalstyletypeid'].value_counts()




dfprop['basementsqft'].max()




dfprop['basementsqft'].min()




plt.hist(dfprop.basementsqft.dropna())
plt.show()




dfprop.bathroomcnt.head()




np.sum(np.isnan(dfprop.bathroomcnt))




np.sum(dfprop.bathroomcnt == 0)




dfprop.bathroomcnt.count()




plt.hist(dfprop.bathroomcnt.dropna())
plt.show()




dfprop.bedroomcnt.head()




dfprop.bedroomcnt.tail()




dfprop.fips.head()




dfprop.fips.tail()




dfprop.fips.value_counts()




dfprop.fullbathcnt.value_counts()




dfprop.propertylandusetypeid.value_counts()




dfprop.assessmentyear.value_counts()




dfprop.landtaxvaluedollarcnt.head()




dfprop.landtaxvaluedollarcnt.tail()




dfprop.landtaxvaluedollarcnt.describe()




tax_mask = pd.isnull(dfprop.structuretaxvaluedollarcnt) == False
dfprop.taxvaluedollarcnt[tax_mask] - dfprop.landtaxvaluedollarcnt[tax_mask] - dfprop.structuretaxvaluedollarcnt[tax_mask]




dfprop.taxvaluedollarcnt - dfprop.taxamount




dfprop.taxamount.describe()




tvdc_mask = pd.isnull(dfprop.taxvaluedollarcnt) == False
ta_mask = pd.isnull(dfprop.taxamount) == False

ta_tvdc_mask = ta_mask & tvdc_mask




plt.scatter(dfprop.taxamount[ta_tvdc_mask],dfprop.taxvaluedollarcnt[ta_tvdc_mask])
plt.show()




predictors = ['bathroomcnt',              'bedroomcnt',              'calculatedbathnbr',              'calculatedfinishedsquarefeet',              'fullbathcnt',              'latitude',              'longitude',              'roomcnt',              'yearbuilt',              'structuretaxvaluedollarcnt',              'landtaxvaluedollarcnt',              'regionidcounty']
response = ['logerror']

              




cols = predictors + response
cols




df_clean = dfcomb_nar.loc[:,cols].dropna()




df_clean.shape




df_clean.to_csv('version_1.csv')

