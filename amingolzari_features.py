#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np




t = 732
m=8
n=8




df = pd.read_csv('../input/data.txt',skiprows=2,sep=' ',names=list(map(str,(list(range(n))))))




arr = np.array(df)




td = arr.reshape(732,8,8)




# new features




df_r = pd.DataFrame(columns=['id','m1','m2','m3','m4'])
for i in range(m):
    for j in range(n):
        for k in range(732):
            temp = td[::24,i,j]
            temp = temp[temp!=-1]
            temp2 = td[k,:,:]
            temp2 = temp2[temp2!=-1]
            temp3 = td[::24*7,i,j]
            temp3 = temp3[temp3!=-1]
            x = i//4
            y = j//4
            temp4 = td[k,x*4:(x+1)*4,y*4:(y+1)*4]
            temp4 = temp4[temp4!=-1]
            st = str(k)+':'+str(i)+':'+str(j)
            df_r = df_r.append({'id':st,'m1':temp.mean(),'m2':temp2.mean(),'m3':temp3.mean(),'m4':temp4.mean()},ignore_index=True)




new_df = df.stack()




new_df = pd.DataFrame(new_df)




new_df.columns=['value']




new_df.reset_index(inplace=True)




new_df.columns= ['t','n','val']




new_df['m'] = new_df['t'].apply(lambda x : int(x)%8)




new_df['t'] = new_df['t'].apply(lambda x:x//8)




new_df['hour'] = new_df['t'].apply(lambda x : x%24 )




new_df['day'] = new_df['t'].apply(lambda x : x//24 )




new_df['mm']=new_df['m']
new_df['nn']= new_df['n']




new_df['id'] = new_df['t'].map(str)+':'+new_df['mm'].map(str)+':'+new_df['nn'].map(str)




new_df = df_r.set_index('id').join(new_df.set_index('id'))




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(new_df.drop(['val','t','mm','nn'],axis=1))




scaler.transform(new_df.drop(['val','t','mm','nn'],axis=1))




new_df[['m1','m2','m3','m4','n','m','hour','day']]=scaler.transform(new_df.drop(['val','t','mm','nn'],axis=1))




train = new_df[new_df['val'] !=-1]




X_train = train.drop(['val','t','mm','nn'], axis=1)
y_train = train['val'].values




test = new_df[new_df['val'] == -1]




X_test = test.drop(['val','t','mm','nn'], axis=1)




import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam




model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(8,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1))
model.summary()




from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 




model.compile(loss=root_mean_squared_error, optimizer='adam')




model.fit(X_train, y_train, batch_size=128, epochs=240, verbose=1,validation_split=0.2)




from xgboost import XGBRegressor
model_XGB = XGBRegressor()
model_XGB.fit(X_train,y_train)




pd.Series(model_XGB.feature_importances_,index=list(X_train.columns.values)).sort_values(ascending=True).plot(kind='barh',figsize=(12,18),title='XGBOOST FEATURE IMPORTANCE')




def make_positive(x):
    if x<0:
        return 0
    else:
        return x




predict = model.predict(X_test)




X_test['pred']= predict




X_test.reset_index(inplace=True)




X_test['demand'] = X_test['pred'].apply(make_positive)




X_test[['id','demand']].to_csv('result.csv',index=False)

