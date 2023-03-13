#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
sub = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
sub.to_csv('submission.csv',index = 'false')




train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')




test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')




train_labels = pd.read_csv ('/kaggle/input/data-science-bowl-2019/train_labels.csv')




submission = pd.read_csv ('/kaggle/input/data-science-bowl-2019/sample_submission.csv')




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler 

import plotly.express as pe
import seaborn as sns
sns.set()




train





test




desc_test = test.describe() 
desc_test




desc_train = train.describe() 
desc_train




train.info()




test.info()




train.isnull()




test.isnull()




def chk_corr(df):
    corrs = train.corr()
    plt.figure(figsize = (7,7))
    # Heatmap of correlations
    sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    plt.title('Correlation Heatmap');




chk_corr(train)




def chk_corr(df):
    corrs = test.corr()
    plt.figure(figsize = (7,7))
    # Heatmap of correlations
    sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
    plt.title('Correlation Heatmap');




chk_corr(test)




#Import label encoder
from sklearn import preprocessing

#Create label encoder
le = preprocessing.LabelEncoder()




#Merge train and train_lables dataframes
group = pd.merge(train, train_labels, on= ['game_session', 'installation_id', 'title']).sort_values(['game_session','installation_id'])
train.head()




#Condense rows into summarized dataframe
group.groupby(['game_session', 'installation_id', 'title', 'type', 'world', 'num_correct', 'num_incorrect','accuracy_group']).agg({'game_time':'sum', 'event_code':list})




# First Feature
#group['title'] = group['title'].astype('category').cat.codes
title = le.fit_transform(group['title'])

#Second Feature
game_time = le.fit_transform(group['game_time'])
#Second Feature
world = le.fit_transform(group['world'])
#Second Feature
game_session = le.fit_transform(group['game_session'])
                                   
#Target variable from training labels
outcome = le.fit_transform(group['accuracy_group'])




features = list(zip(world,game_time))






X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size = .8)





sc = StandardScaler() 
  
X_train1 = sc.fit_transform(X_train) 
X_test = sc.transform(X_test) 




# Applying PCA function on training 
# and testing set of X component 
from sklearn.decomposition import PCA 
  
pca = PCA(n_components = 2) 
  
X_train = pca.fit_transform(X_train) 
X_test = pca.transform(X_test) 
  
explained_variance = pca.explained_variance_ratio_ 

pca






model = KNeighborsClassifier(n_neighbors=3)


model.fit(X_train,y_train)




predicted = model.predict(X_test)
predicted




final_model =  DecisionTreeClassifier(max_depth=10)

final_model.fit(X_train,y_train)




# Fitting Logistic Regression To the training set 
from sklearn.linear_model import LogisticRegression 

classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 




# Predicting the test set result using 
# predict function under LogisticRegression 
y_pred = classifier.predict(X_test)
y_actu = classifier.predict(X_test)




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from mpl_toolkits.axes_grid1 import make_axes_locatable




from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
f1 = f1_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred,  average='micro')
roc_auc = accuracy_score(y_test, y_pred)
print(f"F Score: {f1}")
print(f"Precision Score: {precision}")
print(f"Recall Score: {recall}")
print(f"ROC-AOC Score: {roc_auc}")




type_count = test['type'].value_counts().reset_index()
total = len(test)
type_count['percent'] = round((type_count['type']/total)*100,2)
print(type_count)

fig = pe.bar(type_count, x='index', y='type',
             hover_data=['index','percent'], color='type',
             labels={'type':'Type Count'}, height=400)
fig.show()




type_count = train['type'].value_counts().reset_index()
total = len(train)
type_count['percent'] = round((type_count['type']/total)*100,2)
print(type_count)

fig = pe.bar(type_count, x='index', y='type',
             hover_data=['index','percent'], color='type',
             labels={'type':'Type Count'}, height=400)
fig.show()




# Exploring Test Dataset.
test.shape
title_ins = test.groupby('title')['game_session'].count().reset_index()
title_ins.sort_values(by=['game_session'], inplace = True, ascending = False)
print(title_ins[0:10]) #this will print the top 10

fig = pe.bar(title_ins[0:10], x='title', y='game_session', #make a chart of the top 10 using game session
             hover_data=['title'], color='title',
             labels={'game_session':'game session Count'}, height=400, title = 'Highly played Games using test dataset')
fig.show()




import statsmodels.api as sm

X_train_2 = sm.add_constant(X_train) 
est = sm.OLS(y_train, X_train_2)
est2 = est.fit()

print("summary()\n",est2.summary())

