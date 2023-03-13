#!/usr/bin/env python
# coding: utf-8





import keras
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cross_validation import train_test_split

# Using TensorFlow backend.




TEST_SIZE = 0.1




X = np.load('../input/train.csv')
y = np.load('../input/train.csv')




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=11)






from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, GRU
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint






model = Sequential([
    Dense(
        1024, 
        input_dim=X_train.shape[1], 
        activation='relu'
    ),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(loss='mae', optimizer='RMSprop')




hist = model.fit(
        X_train,
        y_train,
        batch_size=32,
        nb_epoch=10, 
        validation_data=(X_test, y_test),
        callbacks = [
            ModelCheckpoint(
                "model-{epoch:03d}@l={loss:.5f},vl={val_loss:.5f}.h5",
                monitor='val_loss', 
                verbose=False, 
                save_best_only=False,
                mode='max'
            )
        ]
    )

