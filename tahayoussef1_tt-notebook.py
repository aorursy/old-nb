#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

def corr_plot2(dataframe2, top_n2, target2, fig_x2, fig_y2):
    corrmat2 = dataframe2.corr()
    #top_n - top n correlations +1 since price is included
    top_n2 = top_n2 + 1 
    cols2 = corrmat2.nlargest(top_n2, target2)[target2].index
    cm2 = np.corrcoef(donnees[cols2].values.T)
    f2, ax2 = plt.subplots(figsize=(fig_x2,fig_y2))
    sns.set(font_scale=1.25)
    cmap2 = plt.cm.viridis
    hm2 = sns.heatmap(cm2, cbar=False, annot=True, square=True,cmap = cmap2, fmt='.2f', annot_kws={'size': 10},	 yticklabels=cols2.values, xticklabels=cols2.values)
    plt.show()
    return cols2,cm2
def corr_plot(dataframe, top_n, target, fig_x, fig_y):
    corrmat = dataframe.corr()
    #top_n - top n correlations +1 since price is included
    top_n = top_n + 1 
    cols = corrmat.nlargest(top_n, target)[target].index
    cm = np.corrcoef(donnees[cols].values.T)
    """f, ax = plt.subplots(figsize=(fig_x,fig_y))
    sns.set(font_scale=1.25)
    cmap = plt.cm.viridis
    hm = sns.heatmap(cm, cbar=False, annot=True, square=True,cmap = cmap, fmt='.2f', annot_kws={'size': 10},	 yticklabels=cols.values, xticklabels=cols.values)
    plt.show()"""
    return cols,cm


donnees1 = pd.read_csv("train.csv")
donnees2 = pd.read_csv("macro.csv")
donnees = pd.merge(donnees1, donnees2, on='timestamp')
del donnees["timestamp"]


#missing data
total = donnees.isnull().sum().sort_values(ascending=False) #calculer le total des valeurs manquantes
percent = (donnees.isnull().sum()/donnees.isnull().count()).sort_values(ascending=False) #le pourcentage de valeur manquantes pour chaque variable 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
donnees = donnees.drop((missing_data[missing_data['Total'] > 10000]).index,1)
#print donnees.shape
donnees = donnees.dropna(thresh=donnees.shape[1])
print donnees.shape
dtype_df = donnees.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
tab = []
for x in range(0,dtype_df.shape[0]):
	if(dtype_df["Column Type"][x] == "object"):
		tab.append(dtype_df["Count"][x])
		
for x in range(0,15):
	donnees[tab[x]] = pd.factorize(donnees[tab[x]])[0]
corr_20,cm = corr_plot(donnees, 150, 'price_doc', 10,10)

corr_20 = corr_20[0:35]
print corr_20
corr_22 = corr_plot2(donnees, 10, 'price_doc', 10,10)
donnees = donnees[corr_20].copy()
test = pd.read_csv("test.csv")
test = pd.merge(test, donnees2, on='timestamp')
del test["timestamp"]
print test.shape
test = test[corr_20].copy()


#missing data
total = test.isnull().sum().sort_values(ascending=False) #calculer le total des valeurs manquantes
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False) #le pourcentage de valeur manquantes pour chaque variable 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
test  = test.fillna(test.mean())
total = test.isnull().sum().sort_values(ascending=False) #calculer le total des valeurs manquantes
percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False) #le pourcentage de valeur manquantes pour chaque variable 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


total = donnees.isnull().sum().sort_values(ascending=False) #calculer le total des valeurs manquantes
percent = (donnees.isnull().sum()/donnees.isnull().count()).sort_values(ascending=False) #le pourcentage de valeur manquantes pour chaque variable 
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
donnees = donnees.dropna(thresh=donnees.shape[1])
dtype_df = donnees.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
tab = []
for x in range(0,dtype_df.shape[0]):
    if(dtype_df["Column Type"][x] == "object"):
        tab.append(dtype_df["Count"][x])
        #print dtype_df["Count"][x]

for x in range(0,len(tab)):
    donnees[tab[x]] = pd.factorize(donnees[tab[x]])[0]
    test[tab[x]] = pd.factorize(test[tab[x]])[0]

price = donnees.price_doc
test = test.drop('price_doc', 1)
donnees = donnees.drop('price_doc', 1)
print(test.columns)
print('--------------------------------')
print(donnees.columns)
#----------------------------------------
modeleReg1=LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(donnees, price, train_size=0.9)
modeleReg1.fit(X_train,y_train) #effectuer la regression lineaire
y_predicted2 = modeleReg1.predict(X_test)
R = modeleReg1.score(X_test,y_test)
print "R = ",R
ms = sqrt(mean_squared_error(y_test, y_predicted2))
print(ms)

y_predicted = modeleReg1.predict(test)
id_test = range(30474,38136)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predicted})
output.head()
output.to_csv('taha.csv', index=False)

