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

# Any results you write to the current directory are saved as output.




from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py




from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.utils import shuffle
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense




batch_size = 32

data_root_dir = '../input/dogs-vs-cats-redux-kernels-edition/'

keras_models_dir = '../input/keras-models/'

print(check_output(["ls", keras_models_dir]).decode("utf8"))
print(check_output(["ls", data_root_dir]).decode("utf8"))




img_width, img_height = 224, 224

vgg16 = applications.VGG16(include_top=False, weights=None) #input_shape = (3, img_width, img_height)

def load_split_weights(model, model_path_pattern='model_%d.h5', memb_size=102400000):  
    model_f = h5py.File(model_path_pattern, "r", driver="family", memb_size=memb_size)
    topology.load_weights_from_hdf5_group_by_name(model_f, model.layers)
    return model

model_path_pattern = keras_models_dir + "vgg16_weights_tf_dim_ordering_tf_kernels_%d.h5" 
vgg16 = load_split_weights(vgg16, model_path_pattern = model_path_pattern)

# set the first 25 layers (up to the last conv block) to non-trainable (weights will not be updated)
for layer in vgg16.layers[:25]:
    layer.trainable = False

x = vgg16.get_layer('block5_conv3').output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=vgg16.input, outputs=x)

model.summary()




model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




def augment(src, choice):
    if choice == 0:
        # Rotate 90
        src = np.rot90(src, 1)
    if choice == 1:
        # flip vertically
        src = np.flipud(src)
    if choice == 2:
        # Rotate 180
        src = np.rot90(src, 2)
    if choice == 3:
        # flip horizontally
        src = np.fliplr(src)
    if choice == 4:
        # Rotate 90 counter-clockwise
        src = np.rot90(src, 3)
    if choice == 5:
        # Rotate 180 and flip horizontally
        src = np.rot90(src, 2)
        src = np.fliplr(src)
    return src




import glob
from sklearn.cross_validation import train_test_split
from numpy import random

train_dogs = glob.glob(data_root_dir + "train/dog.*")
train_cats = glob.glob(data_root_dir + "train/cat.*")
#print (train_cats[:1])

# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
images = train_dogs[:1200] + train_cats[:1200]
random.shuffle(images)

#print(images[:2])
labels = []
for i in images:
    #print(i)
    if "dog." in i:
        labels.append(1)
    else:
        labels.append(0)
        
batch_size = 10

def process_img(i):
    img = load_img(dogs[i])  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    #x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    return x;

train_images, validation_images = train_test_split(images, test_size=0.4)

#print("Train shape: {}".format(train.shape))
#print("Test shape: {}".format(test.shape))
#print("Validation shape: {}".format(valid.shape))
#print(len(train_images))
print(train_images[:1])
print(validation_images[:1])




import matplotlib.pyplot as plt
from scipy.misc import imresize

def imread(image_path):
    image = plt.imread(image_path)
    image = imresize(image, (img_width, img_height))
    return image

def preprocess_images(image_names, seed, datagen):
    #print (image_names)
    np.random.seed(seed)
    X = np.zeros((len(image_names), img_width, img_height, 3))
    for i, image_name in enumerate(image_names):
        #print (image_name)
        image = imread(image_name)
        X[i] = datagen.random_transform(image)
    return X

def image_triple_generator(train_images, batch_size):
    datagen_args = dict(rotation_range=10,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True)
    datagen = ImageDataGenerator(**datagen_args)
    
    while True:
        # loop once per epoch
        num_recs = len(train_images)
        #print(num_recs)
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            # loop once per batch
            batch_indices = indices[bid * batch_size : (bid + 1) * batch_size]
            #print(batch_indices)
            batch = [train_images[i] for i in batch_indices]
            #print(batch)
            # make sure image data generators generate same transformations
            seed = np.random.randint(low=0, high=1000, size=1)[0]
            batch_label = []
            batch_img = preprocess_images(batch, seed, datagen)
            for i in batch:
                if "dog." in i:
                    batch_label.append(1)
                else:
                    batch_label.append(0)
                    
            yield batch_img, batch_label
            
batches = image_triple_generator(train_images, 4)
val_batches = image_triple_generator(validation_images, 4)




trn_classes = len(train_images)
val_classes = len(validation_images)
steps_per_epoch=int(np.ceil(trn_classes/batch_size))
validation_steps=int(np.ceil(val_classes/batch_size))    
epochs=15

print ("epochs:" + str(epochs))
print ("batch_size:" + str(batch_size))
print ("trn_classes:" + str(trn_classes))
print ("val_classes:" + str(val_classes))
print ("steps_per_epoch:" + str(steps_per_epoch))
print ("validation_steps:" + str(validation_steps))




from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import LearningRateScheduler
import os.path

fine_weights_path = 'tune_weights.h5'

if os.path.isfile(fine_weights_path) :
    print ("load fine_weights_path:" + fine_weights_path)
    model.load_weights(fine_weights_path)
    
def step_decay(epoch):
    if epoch >= 0 and epoch < 2:
        lrate = 0.0001 #Default Adam lr=0.001
    elif epoch >= 2 and epoch < 5:
        lrate = 0.00001
    elif epoch >= 5 and epoch < 10:
        lrate = 0.000001
    elif epoch >= 15 and epoch < 20:
        lrate = 0.0000001
    else:
        lrate = 0.0000001
    
    print (str(epoch) + " learning rate:%.6f" % lrate)
    return lrate

reduce_lr = LearningRateScheduler(step_decay)

callbacks_list = [
    ModelCheckpoint(fine_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_acc', patience=5, verbose=1),reduce_lr]




history = model.fit_generator(batches, 
                    steps_per_epoch=steps_per_epoch, 
                    epochs=epochs, 
                    validation_data=val_batches, 
                    validation_steps=validation_steps,
                    callbacks=callbacks_list,          
                    verbose=1)




import seaborn

seaborn.countplot(labels)
#seaborn.plt.title('Cats and Dogs')




print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*history.history['acc'][-1], 100*history.history['val_acc'][-1]))




model.save_weights(fine_weights_path)




import matplotlib.pyplot as plt

# list all data in history
print(history.history.keys())

plt.plot(history.history['val_acc'])
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()






