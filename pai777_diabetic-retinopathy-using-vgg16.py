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




df_train=pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
df_test=pd.read_csv('/kaggle/input/aptos2019-blindness-detection/test.csv')                   




df_test.head()




df_train.shape
import cv2




def preprocess(imag_path):
    imag=cv2.imread(imag_path)
    imag=cv2.resize(imag,(150,150))
    return imag





x_train=np.empty((3662,150,150,3))

image_id=df_train['id_code']
from tqdm import tqdm




for i,image_id in enumerate(tqdm(df_train['id_code'])):
    x_train[i,:,:,:]=preprocess(f'../input/aptos2019-blindness-detection/train_images/{image_id}.png')




df_test.shape
image_id=df_test['id_code']




x_test=np.empty((1928,150,150,3))
for i,image_id in enumerate(tqdm(df_test['id_code'])):
    x_test[i,:,:,:]=preprocess(f'../input/aptos2019-blindness-detection/test_images/{image_id}.png')




import matplotlib.pyplot as plt
import matplotlib.image as mpimg




x= df_train.loc[:,'id_code']
y_test=df_test




y_test




image=cv2.imread(f'../input/aptos2019-blindness-detection/train_images/{x[0]}.png')




plt.imshow(image)




image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)




plt.imshow(image)




image=cv2.resize(image,(150,150))




import keras
plt.imshow(image)
y_train=df_train['diagnosis']
y_train = keras.utils.to_categorical(y_train, 5)




from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=0.15, 
    
)




from keras.models import Sequential
from keras.layers import Dense, Activation,Conv2D,MaxPool2D,Dropout,Flatten,Activation
from keras.preprocessing.image import ImageDataGenerator




model=Sequential()




model.add(Conv2D(64,(3,3),padding='valid',input_shape=(150,150,3),activation='relu'))




model.summary()




model.add(Conv2D(64,(3,3),padding='valid',activation='relu'))




model.add(MaxPool2D(2,2))




model.add(Conv2D(128,(3,3),padding='valid',activation='relu'))




model.add(Conv2D(128,(3,3),padding='valid',activation='relu'))




model.add(MaxPool2D(2,2))




model.add(Conv2D(256,(3,3),padding='valid',activation='relu'))
model.add(Conv2D(256,(3,3),padding='valid',activation='relu'))

model.add(MaxPool2D(2,2))




model.add(Flatten())




model.add(Dense(128,activation='relu'))




model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))




model.add(Dense(5,activation='softmax'))




model.summary()




model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])






def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.15,  
        fill_mode='constant',
        cval=0.,  
        horizontal_flip=True,  
        vertical_flip=True,  
    )


data_generator = create_datagen().flow(x_train, y_train, batch_size=32)




model.fit_generator(data_generator,epochs=5,steps_per_epoch=x_train.shape[0]/32,validation_data=(x_val, y_val))




x_test.shape




x_test[0]




y_final=model.predict(x_test)




y_final[0]




np.argmax(y_final[1817])




y_final

