#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#from sklearn.ensemble import RandomForestClassifier

#keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils

# Any results you write to the current directory are saved as output.




# define baseline model
def baseline_model():
	# create model
    model = Sequential()
    model.add(Dense(8, input_dim=93, activation='relu'))
    model.add(Dense(9, activation='softmax'))
	# Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model




dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[: ,1 :-1].values
Y = dataset.iloc[:, -1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN" , strategy = "mean", axis =0)
imputer = imputer.fit(X[:,:])
X[:,:] = imputer.transform(X[:,:])

from sklearn.preprocessing import LabelEncoder
label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

from sklearn.cross_validation import train_test_split
X_trn, X_test, Y_trn , Y_test = train_test_split(X,Y, test_size= 0.2, random_state= 0)

#keras
knn = KerasClassifier(build_fn=baseline_model, epochs=1000, batch_size=5, verbose=0)
knn.fit(X_trn, Y_trn)

# Random forest
#knn = RandomForestClassifier(n_jobs=10, random_state=36)
#knn.fit(X_trn, Y_trn)
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors = 10, metric= 'minkowski', p=2)
#knn.fit(X_trn, Y_trn)

Y_pred = knn.predict(X_test)




print(Y_pred)




dataset_test = pd.read_csv('../input/test.csv')
X_test_out = dataset_test.iloc[:, 1:].values
#print(X_test_out)
Y_pred_out = knn.predict_proba(X_test_out)
#Y_pred_out = knn.predict(X_test_out)




#Y_pred_out = Y_pred_out.transpose()




submission = pd.DataFrame({ "id": dataset_test["id"]})

i = 0

# Create column name based on target values(see sample_submission.csv)
range_of_classes = range(1, 10)
for num in range_of_classes:
    col_name = str("Class_{}".format(num))
    submission[col_name] = Y_pred_out[:,i]
    i = i + 1
submission.to_csv('otto.csv', index=False)    




# """"l = np.array(Y_pred_out)
# print(l)
# #b = np.zeros((144368,9))
# #a = pd.DataFrame(b, columns= ['Class_1', 'Class_2' ,'Class_3', 'Class_4', 'Class_5','Class_6','Class_7','Class_8','Class_9'])
# #a = a.astype(int)
# #a.insert(0, 'ID',range(1,1+len(a)))
# #a.head()
# #
# #index=0;
# #for i in np.nditer(l):
#     #print(i)
#  #   a[str('Class_{}'.format(i+1))][index] = 1
#     #print('row=',index, "  col=",str('Class_{}'.format(i+1)))
#   #  index = index+1;
#     #a[ : , i] = i + 1""""




#a.head()




#submission = pd.read_csv('../input/sampleSubmission.csv')
#submisssion = a
#submission.head()




#submission.to_csv('otto1.csv', index=False)




#a = np.arange(Y_pred_out).reshape(9, 144368)




#Y_pred_out = pd.DataFrame(Y_pred_out, index=sample.id.values, columns=sample.columns[1:])




#sample= pd.read_csv('../input/sampleSubmission.csv')




#dataset_submission.head()




#dataset.target.unique()




#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(Y_pred, Y_test)




# Visualising the Training set results
#from matplotlib.colors import ListedColormap
#X_set, Y_set = X_trn, Y_trn
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                    #  np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
 #            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('K-NN (Training set)')
#plt.xlabel('prediction')
#plt.ylabel('class')
#plt.legend()
#plt.show()




# """"# Visualising the Test set results
# from matplotlib.colors import ListedColormap
# X_set, Y_set = X_test, Y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('K-NN (Test set)')
# plt.xlabel('prediction')
# plt.ylabel('class')
# plt.legend()
# plt.show() """"









print(X_test)




print(X_test_out)






