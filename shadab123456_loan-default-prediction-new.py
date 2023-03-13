#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer #imputing  missing values

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.




data = pd.read_csv("/kaggle/input/loan-default-prediction/train_v2.csv.zip")




#Shape of data
data.shape




#Information of data for data types 
data.info()




# Extract the features which has categorical values

data_categorical = data.select_dtypes(include=['object'])
(data_categorical.head())




print(data_categorical.columns)




# Count the frequency of each class or labels of a categorical data
print(data_categorical['f137'].value_counts())
print(data_categorical['f138'].value_counts())
print(data_categorical['f206'].value_counts())
print(data_categorical['f207'].value_counts())
print(data_categorical['f276'].value_counts())
print(data_categorical['f277'].value_counts())
print(data_categorical['f338'].value_counts())
print(data_categorical['f390'].value_counts())
print(data_categorical['f391'].value_counts())
print(data_categorical['f419'].value_counts())
print(data_categorical['f420'].value_counts())
print(data_categorical['f469'].value_counts())
print(data_categorical['f472'].value_counts())
print(data_categorical['f534'].value_counts())
print(data_categorical['f537'].value_counts())
print(data_categorical['f626'].value_counts())
print(data_categorical['f627'].value_counts())
print(data_categorical['f695'].value_counts())
print(data_categorical['f698'].value_counts())




data.describe()




data_missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
data_missing.head()




#Features with missing values for categorical values
data_categorical.loc[1:5:, data_categorical.isnull().any()]




# Filling each missing categorical data with mode
data_categorical_imputed = data_categorical.fillna(data_categorical.mode().iloc[0])





data_categorical_imputed.isnull().sum()




# Extracting data with float and integer data types 
data_num = data.select_dtypes(include=['float64','int64'])
(data_num.head())




#Features with missing values for numerical values
data_num.loc[1:5:, data_num.isnull().any()]




# imputing missing terms of numerical data with mean
from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(data_num)
data_num_imputed =pd.DataFrame(imputer.transform(data_num))




data_num_imputed.columns = data_num.columns




data_num_imputed.isnull().sum()




#removing id value from data
data_num_imputed.drop('id', axis=1, inplace=True)




X_data_num   =  data_num_imputed.iloc[:,0:750]




#Threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = X_data_num.corr().abs()
corr_matrix.head()




upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()




# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))




X_data_num_1 = X_data_num.drop(columns = to_drop)




print("shape of original numerical feature", X_data_num.shape)
print("shape of reduced numerical feature",X_data_num_1.shape)




# # Correlations between Features and Target

# Find all correlations and sort 
correlations_data = data_num_imputed.corr()['loss'].sort_values()

# Print the most negative correlations
print(correlations_data.head(30), '\n')

# Print the most positive correlations
print(correlations_data.tail(30))




# Find the column which hsa constant terms due to correlation to target is becoming 0
print([column for column in X_data_num_1.columns if len(X_data_num_1[column].unique())==1])




# Drop columns with correlations of NaN
to_drop_1 = [column for column in X_data_num_1.columns if len(X_data_num_1[column].unique())==1]
print('There are %d columns to remove.' % (len(to_drop_1)))
X_data_num_2 = X_data_num_1.drop(columns = to_drop_1)
print("shape of new numerical feature is :",X_data_num_2.shape)




x= data_categorical_imputed
y= data_num_imputed['loss']




y.value_counts()




# making binary problem 

y[y>0]=1




y.value_counts()




# modeling 

from sklearn.ensemble import RandomForestClassifier
# utilities
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# memory management
import gc




# Initialize an empty array to hold feature importances
feature_importances = np.zeros(X_data_num_2 .shape[1])




# Create the model with several hyperparameters
model_random = RandomForestClassifier(n_jobs=-1, n_estimators=250)




# Fit the model twice to avoid overfitting
for i in range(2):
    
    # Split into training and validation set
    train_features, valid_features, train_y, valid_y = train_test_split(X_data_num_2, y, test_size = 0.25, random_state = i)
    
    # Train the random forest model
    model_random.fit(train_features, train_y)
    
    # Record the feature importances
    feature_importances += model_random.feature_importances_




# average of feature importances
feature_importances = feature_importances / 2
feature_importances = pd.DataFrame({'feature': list(X_data_num_2.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)




feature_importances 




#Plotting top 20 features
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
figure(num=None, figsize=(18, 6), dpi=80, facecolor='w', edgecolor='k')
plt.bar(feature_importances['feature'][0:20], feature_importances['importance'][0:20], width=0.8, bottom=None, align='center', data=feature_importances)




# Find the features with least importance with threshold less than .0020
least_important_features = list(feature_importances[feature_importances['importance'] < 0.0020]['feature'])
print('There are %d features with least importance' % len(least_important_features ))
feature_importances.tail()




# drop the columns with least importance
X_data_num_3 = X_data_num_2.drop(columns = least_important_features)




X_data_num_3.head()




# Make features and target as X and Y
X = X_data_num_3.copy()
Y = y.copy()




#Train a model with these features
#Logistic Regression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=42)




# # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)




# Convert y to one-dimensional array (vector)
y_train = np.array(y_train).reshape((-1, ))
y_test = np.array(y_test).reshape((-1, ))





# We will compare 2 different machine learning Cassification models:

# 1 - Logistic Regression
# 2 - Random Forest Classification




from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=10, random_state=0).fit(X_train, y_train)




y_pred = clf.predict(x_test)




print('Logistic Performance on the training set: Cross Validation Score = %0.4f'%round((clf.score(X_train,y_train)).mean(),4))




print('Logistic Performance on the validation set: Cross Validation Score',round((clf.score(X_test,y_test)),4))




import seaborn as sns
import matplotlib.pyplot as plt     
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
class_label = ["0", "1"]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
sns.heatmap(df_cm, annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()




from sklearn.metrics import f1_score




f_score_logistic_test =  round(f1_score(y_test, y_pred, average='macro'),2)




print("F score from logistic regression for test data is :",f_score_logistic_test)




from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

random = RandomForestClassifier(n_estimators =15) 
# 10-Fold Cross validation
scores  = cross_val_score(random, X_train, y_train, cv=10)




ranmodel  = random.fit( X_train, y_train)




y_ran_pred  = ranmodel.predict(X_test)




print('Random Forest Performance on the training  set: Cross Validation Score',round(scores.mean(),4))




import seaborn as sns
import matplotlib.pyplot as plt     
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_ran_pred)
class_label = ["0", "1"]
df_cm1 = pd.DataFrame(cm1, index=class_label,columns=class_label)
sns.heatmap(df_cm1, annot=True,fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()




print("F score from Random Forest for test data is :",round(f1_score(y_test, y_ran_pred, average='macro'),3))




test_data = pd.read_csv("/kaggle/input/loan-default-prediction/test_v2.csv.zip")




test_data.head()




test_data.shape




test_data.isnull().sum()




X.columns




feature_test  = test_data[X.columns]




feature_test.shape




feature_test.isnull().sum()




# imputing missing terms of numerical data with mean
from sklearn.impute import SimpleImputer 
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(feature_test)
test_features_imputed =pd.DataFrame(imputer.transform(feature_test))




test_features_imputed.isnull().sum()




# # Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features_test_scaled = sc.fit_transform(test_features_imputed)




Predicted_values = ranmodel.predict(features_test_scaled)




Predicted_values_df = pd.DataFrame({'default':Predicted_values})




sample_submission  = pd.read_csv("/kaggle/input/loan-default-prediction/sampleSubmission.csv")




sample_submission.head()




sample_submission.id = test_data.id




sample_submission.loss = Predicted_values_df.default




submission = sample_submission.copy()




submission.head() 




Submission = submission.to_csv(index=False)




import os 
os.chdir(r'/kaggle/working')





submission.to_csv(r'SUBMISSION.csv',index=False)




from IPython.display import FileLink
FileLink(r'SUBMISSION.csv')

