#!/usr/bin/env python
# coding: utf-8



# the following are the important Package used for this EDA process

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot
from wordcloud import WordCloud
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.plotly as py
from plotly import tools
from datetime import date
import seaborn as sns
import random 
import warnings
warnings.filterwarnings("ignore")
import matplotlib as matplot
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




application_train = pd.read_csv('../input/application_train.csv')
application_test= pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')




print("The number of Features in application train dataset :",application_train.shape[1])
print("The number of Rows in application Train dataset :",application_train.shape[0])




application_train.head()




def type_features(data):
    categorical_features = data.select_dtypes(include = ["object"]).columns
    numerical_features = data.select_dtypes(exclude = ["object"]).columns
    print( "categorical_features :",categorical_features)
    print('-----'*40)
    print("numerical_features:",numerical_features)
    




type_features(application_train)





def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms= ms[ms["Percent"] > 0]
    f,ax =plt.subplots(figsize=(15,10))
    plt.xticks(rotation='90')
    fig=sns.barplot(ms.index, ms["Percent"],color="green",alpha=0.8)
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title('Percent missing data by feature', fontsize=15)
    #ms= ms[ms["Percent"] > 0]
    return ms




missingdata(application_test)




application_test.head(7)




print("the number columns in the application_test dataset",application_test.shape[1])
print("the number rows in application_test dataset",application_test.shape[0])




type_features(application_test)




missingdata(application_test)




bureau.head(6)




print("The number of features is :",bureau.shape[1],"The number of row is:",bureau.shape[0])




type_features(bureau)




missingdata(bureau)




bureau_balance.head(7)




print("the number of columns",bureau_balance.shape[1],"the number of rows :",bureau_balance.shape[0])




type_features(bureau_balance)




total = bureau_balance.isnull().sum().sort_values(ascending = False)
percent = (bureau_balance.isnull().sum()/bureau_balance.isnull().count()*100).sort_values(ascending = False)
ms=pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
ms= ms[ms["Percent"] > 0]
ms




POS_CASH_balance.head(6)




POS_CASH_balance.shape




type_features(POS_CASH_balance)




missingdata(POS_CASH_balance)




credit_card_balance.head()




print("the number columns in dataset:",credit_card_balance.shape[1],"The number of rows:",credit_card_balance.shape[0])




type_features(credit_card_balance)




missingdata(credit_card_balance)




previous_application.head(7)




type_features(previous_application)




missingdata(previous_application)








type_features(installments_payments)




missingdata(installments_payments)




f,ax=plt.subplots(1,2,figsize=(12,6))
application_train.TARGET.value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Distribution of target variable')
ax[0].set_ylabel('')
sns.countplot('TARGET',data=application_train,ax=ax[1])
ax[1].set_title('Count of Repayer VS defulter')
plt.show()




def group_by(df,t1='',t2=''):
    a1=df.groupby([t1,t2])[t2].count()
    return a1




def plot_re(df,t1='',t2=''):
    f,ax=plt.subplots(1,2,figsize=(10,6))
    df[[t1,t2]].groupby([t1]).count().plot.bar(ax=ax[0],color='Green')
    ax[0].set_title('count of customer Based on'+t1)
    sns.countplot(t1,hue=t2,data=df,ax=ax[1],palette="spring")
    ax[1].set_title(t1+': Repayer vs Defualter')
    # Rotate x-labels
    plt.xticks(rotation=-90)
    a=plt.show()
    return a




plot_re(application_train,'NAME_EDUCATION_TYPE','TARGET')




plot_re(application_train,'CODE_GENDER','TARGET')




plot_re(application_train,'NAME_INCOME_TYPE','TARGET')




plot_re(application_train,'OCCUPATION_TYPE','TARGET')




plot_re(application_train,'NAME_FAMILY_STATUS','TARGET')




plot_re(application_train,'NAME_HOUSING_TYPE','TARGET')




plot_re(application_train,'NAME_TYPE_SUITE','TARGET')




f, ax = plt.subplots(figsize=(20, 8))
# Employee distri
# Types of colors
color_types = ['#78C850','#F08030','#6890F0','#A8B820','#A8A878','#A040A0','#F8D030',  
                '#E0C068','#EE99AC','#C03028','#F85888','#B8A038','#705898','#98D8D8','#7038F8']

# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='ORGANIZATION_TYPE', data=application_train, palette=color_types).set_title('count based on Organization type');
 
# Rotate x-labels
plt.xticks(rotation=-90)




f, ax = plt.subplots(figsize=(15, 10))
sns.countplot(y="ORGANIZATION_TYPE", hue='TARGET', 
              data=application_train).set_title('REpayer VS Defaulter based on Organization type')




plot_re(application_train,'FLAG_OWN_CAR','TARGET')




plot_re(application_train,'FLAG_OWN_REALTY','TARGET')




plot_re(application_train,'NAME_CONTRACT_TYPE','TARGET')




plot_re(application_train,'WEEKDAY_APPR_PROCESS_START','TARGET')




plot_re(application_train,'HOUSETYPE_MODE','TARGET')




plot_re(application_train,'EMERGENCYSTATE_MODE','TARGET')




sns.set_style('whitegrid')
f, ax = plt.subplots(3,1,figsize=(20,15))

# Types of colors
color_types = ['#78C850','#F08030','#6890F0','#A8B820','#A8A878','#A040A0','#F8D030',  
                '#E0C068','#EE99AC','#C03028','#F85888','#B8A038','#705898','#98D8D8','#7038F8']

# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='CNT_CHILDREN', data=application_train, ax=ax[0],palette=color_types).set_title('count based on Organization type');
sns.countplot("CNT_CHILDREN", hue='TARGET', 
              data=application_train,ax=ax[1]).set_title('REpayer VS Defaulter based on CNT_CHILDREN')
ax[2]=sns.kdeplot(application_train.loc[(application_train['TARGET'] == 0),'CNT_CHILDREN'] , color='b',shade=True,label='NON-PAYERS')
ax[2]=sns.kdeplot(application_train.loc[(application_train['TARGET'] == 1),'CNT_CHILDREN'] , color='r',shade=True, label='REPAYERS')
ax[2].set_title('Children count Distribution - Repayer V.S. Non Repayers')




# Set up the matplotlib figure
f, ax = plt.subplots(2,2, figsize=(15, 10))

# Graph amt annutiy Satisfaction
sns.distplot(application_test.AMT_ANNUITY.dropna(), kde=True, color="g", ax=ax[0,0]).set_title('customer Amount Annual income Distribution')


# Graph amt credit Evaluation
sns.distplot(application_test.AMT_CREDIT.dropna(), kde=True, color="b", ax=ax[0,1]).set_title('customer Amount credit Distribution')

# Graph anaual GOOD PRICE  
sns.distplot(application_test.AMT_GOODS_PRICE.dropna(), kde=True, color="r", ax=ax[1,0]).set_title('customer GOOD PRICE Distribution')


sns.distplot(application_test.AMT_INCOME_TOTAL.dropna(), kde=True, color="y", ax=ax[1,1]).set_title('customer Amount Annual income Distribution')




sns.set_style('whitegrid')
# Set up the matplotlib figure
f, ax = plt.subplots(2,2, figsize=(15, 10))

# Graph amt annutiy Satisfaction
sns.distplot(application_test.DAYS_BIRTH.dropna(), kde=True, color="g", ax=ax[0,0]).set_title('customer Days birth Distribution')

sns.distplot(application_test.DAYS_EMPLOYED.dropna(), kde=True, color="b", ax=ax[0,1]).set_title('customer Employed Distribution')

sns.distplot(application_test.DAYS_ID_PUBLISH.dropna(), kde=True, color="r", ax=ax[1,0]).set_title('customer ID Publish Distribution')


sns.distplot(application_test.DAYS_REGISTRATION.dropna(), kde=True, color="y", ax=ax[1,1]).set_title('customer Days of Registration Distribution')




visual_dat= ['REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']


for i in visual_dat:
    plt.figure(figsize = (10,5))
    plt.subplot(1,3,1)
    sns.countplot(application_train[i])
    plt.subplot(1,3,2)
    sns.countplot(application_train[i],hue=application_train['TARGET'],palette="spring")
    plt.subplot(1,3,3)
    sns.distplot(application_train[i],bins= 20) 
    plt.title(i)
    plt.xticks(rotation=-45)
    plt.show()
    




# Set up the matplotlib figure
f, ax = plt.subplots(ncols=3,figsize=(10, 5))


sns.distplot(application_test.EXT_SOURCE_1.dropna(), kde=True, color="g", 
             ax=ax[0]).set_title('EXT source 1 Distribution')



sns.distplot(application_test.EXT_SOURCE_2.dropna(), kde=True, color="b",
             ax=ax[1]).set_title(' EXT Source 2 Distribution')

 
sns.distplot(application_test.EXT_SOURCE_3.dropna(), kde=True, 
             color="r", ax=ax[2]).set_title('EXt Source 3 Distribution')





val_p=['APARTMENTS_AVG', 'BASEMENTAREA_AVG',
       'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG',
       'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG']
#color_types = ['#78C850','#F08030','#6890F0','#A8B820','#A8A878','#A040A0','#F8D030',  
 #               '#E0C068','#EE99AC','#C03028','#F85888','#B8A038','#705898','#98D8D8','#7038F8']
for i in val_p:
    plt.figure(figsize = (5,5))
    sns.distplot(application_train[i].dropna(), kde=True, color='g')        
    plt.title(i)
    plt.xticks(rotation=-45)
    plt.show()
    




#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(20,15))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        #cbar_kws={'shrink':.9 }, 
        #ax=ax,
        #annot=True, 
        #linewidths=0.1,vmax=1.0, linecolor='white',
        #annot_kws={'fontsize':16}
    )
    
    plt.title('Pearson Correlation of Features')

correlation_heatmap(application_train)




# most correlated features
corrmat = application_train.corr()
top_corr_features = corrmat.index[abs(corrmat["TARGET"])>=0.03]
plt.figure(figsize=(20,10))
g = sns.heatmap(application_train[top_corr_features].corr(),annot=True,cmap="Oranges")




ap_train=application_train
br_data=bureau




print('Applicatoin train shape before merge: ', ap_train.shape)
ap_train = ap_train.merge(br_data, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')
print('Applicatoin train shape after merge: ', ap_train.shape)




plot_re(ap_train,'CREDIT_ACTIVE','TARGET')




plot_re(ap_train,'CREDIT_CURRENCY','TARGET')




plot_re(ap_train,'CREDIT_TYPE','TARGET')




f, ax = plt.subplots(2,3,figsize=(13, 10))


sns.distplot(bureau.DAYS_CREDIT.dropna(), kde=True, color="g", 
             ax=ax[0,0]).set_title('DAYS CREDIT Distribution')



sns.distplot(bureau.CREDIT_DAY_OVERDUE.dropna(), kde=True, color="b",
             ax=ax[0,1]).set_title(' CREDIT DAY OVERDUE Distribution')

 
sns.distplot(bureau.DAYS_CREDIT_UPDATE.dropna(), kde=True, 
             color="r", ax=ax[0,2]).set_title('DAYS CREDIT UPDATE Distribution')

sns.distplot(bureau.AMT_CREDIT_SUM_LIMIT.dropna(), kde=True, color="g", 
             ax=ax[1,0]).set_title(' Distribution')



sns.distplot(bureau.AMT_CREDIT_SUM_DEBT.dropna(), kde=True, color="b",
             ax=ax[1,1]).set_title(' Distribution')

 
sns.distplot(bureau.AMT_CREDIT_SUM_OVERDUE.dropna(), kde=True, 
             color="r", ax=ax[1,2]).set_title('DAYS CREDIT UPDATE Distribution')





f, ax = plt.subplots(figsize=(7,5))

# Types of colors
color_types = ['#78C850','#F08030','#6890F0','#A8B820','#A8A878','#A040A0','#F8D030',  
                '#E0C068','#EE99AC','#C03028','#F85888','#B8A038','#705898','#98D8D8','#7038F8']

# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='STATUS', data=bureau_balance,palette=color_types).set_title('count based on status type')




f, ax = plt.subplots(5,3,figsize=(35,25))

 

sns.countplot(previous_application.NAME_CONTRACT_TYPE.dropna(), palette='spring', 
             ax=ax[0,0]).set_title('Count Distribution')



sns.countplot(previous_application.WEEKDAY_APPR_PROCESS_START.dropna(), palette='spring', 
             ax=ax[0,1]).set_title('Count Distribution')



sns.countplot(previous_application.FLAG_LAST_APPL_PER_CONTRACT.dropna(), palette='spring', 
             ax=ax[0,2]).set_title('Count Distribution')



sns.countplot(previous_application.NAME_CASH_LOAN_PURPOSE.dropna(), palette='spring', 
             ax=ax[1,0]).set_title('Count Distribution')



 
sns.countplot(previous_application.NAME_CONTRACT_STATUS.dropna(), palette='spring', 
             ax=ax[1,1]).set_title('Count Distribution')



sns.countplot(previous_application.NAME_PAYMENT_TYPE.dropna(), palette='spring', 
             ax=ax[1,2]).set_title('Count Distribution')



sns.countplot(previous_application.CODE_REJECT_REASON.dropna(), palette='spring', 
             ax=ax[2,0]).set_title('Count Distribution')



sns.countplot(previous_application.NAME_TYPE_SUITE.dropna(), palette='spring', 
             ax=ax[2,1]).set_title('Count Distribution')



sns.countplot(previous_application.NAME_CLIENT_TYPE.dropna(), palette='spring', 
             ax=ax[2,2]).set_title('Count Distribution')



sns.countplot(previous_application.NAME_GOODS_CATEGORY.dropna(), palette='spring', 
             ax=ax[3,0]).set_title('Count Distribution')


sns.countplot(previous_application.NAME_PORTFOLIO.dropna(), palette='spring', 
             ax=ax[3,1]).set_title('Count Distribution')


sns.countplot(previous_application.NAME_PRODUCT_TYPE.dropna(), palette='spring', 
             ax=ax[3,2]).set_title('Count Distribution')



sns.countplot(previous_application.CHANNEL_TYPE.dropna(), palette='spring', 
             ax=ax[4,0]).set_title('Count Distribution')



sns.countplot(previous_application.NAME_SELLER_INDUSTRY.dropna(), palette='spring', 
             ax=ax[4,1]).set_title('Count Distribution')



sns.countplot(previous_application.NAME_YIELD_GROUP.dropna(), palette='spring', 
             ax=ax[4,2]).set_title('Count Distribution')




("")




val_p=['AMT_ANNUITY',
       'AMT_CREDIT', 'AMT_GOODS_PRICE',
       'HOUR_APPR_PROCESS_START']

for i in val_p:
    plt.figure(figsize = (5,5))
    sns.distplot(application_train[i].dropna(), kde=True, color='g')        
    plt.title(i)
    plt.xticks(rotation=-90)
    plt.show()






