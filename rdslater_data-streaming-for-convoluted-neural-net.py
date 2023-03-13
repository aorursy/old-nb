#!/usr/bin/env python
# coding: utf-8



import os
get_ipython().run_line_magic('mkdir', '"train/dog"')
get_ipython().run_line_magic('mkdir', '"train/cat"')




for dir, subdir, files in os.walk("train"):
    if len(subdir) == 0:
        continue
    for file in files:
        category = file.split(".")[0]
        os.rename("{}/{}".format(dir,file), "{}/{}/{}".format(dir, category, file))




get_ipython().run_line_magic('mkdir', '"valid"')
get_ipython().run_line_magic('mkdir', '"valid/dog"')
get_ipython().run_line_magic('mkdir', '"valid/cat"')




dogs = [x for x in os.listdir("train/dog")]
cats = [x for x in os.listdir("train/cat")]




import random
if len(os.listdir("valid/dog")) < 1:
    for n in random.sample(range(len(dogs)), 1000):
        os.rename("train/dog/{}".format(dogs[n]), "valid/dog/{}".format(dogs[n]))

if len(os.listdir("valid/cat")) < 1:
    for n in random.sample(range(len(cats)), 1000):
        os.rename("train/cat/{}".format(cats[n]), "valid/cat/{}".format(cats[n]))




from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(ROWS,COLS),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'valid',
        target_size=(ROWS, COLS),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

#don't forget, if you leave your images in /test the flow_from directory won't find them
#put test images in /test/<wild name choice here>
#also notice not quotes around the None in the class_mode!!
test_generator=test_datagen.flow_from_directory(
    'test',
     target_size=(ROWS,COLS),
     batch_size=BATCH_SIZE,
     class_mode=None)




from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Dense, Activation, Flatten, Dropout, MaxPooling2D
from keras.regularizers import l2

model = Sequential()
model.add(Convolution2D(16, 4, 4, border_mode='same', input_shape=(ROWS,COLS,CHANNELS),activation='relu'))
model.add(Convolution2D(4, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same'))
model.add(Convolution2D(4, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(4, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same'))
model.add(Flatten())
model.add(Dense(output_dim=64, W_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(output_dim=2, W_regularizer=l2(0.01)))  #binary classification
model.add(Activation('softmax'))




from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, RMSprop
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
model.fit_generator(generator=train_generator,samples_per_epoch=23000,nb_epoch=20)




print("valid set :", model.evaluate_generator(validation_generator,val_samples=2000)[1]*100, "%")
print("--------------------")
print("train set :", model.evaluate_generator(train_generator,val_samples=2000)[1]*100, "%")

