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
import os, cv2, random

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib import ticker

import seaborn as sns




from keras.models import Sequential

from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.utils import np_utils
TRAIN_DIR = '../input/train/'

TEST_DIR = '../input/test/'



ROWS = 64

COLS = 64

CHANNELS = 3



train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset

train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]

train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]



test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]





# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset

train_images = train_dogs[:1000] + train_cats[:1000]

random.shuffle(train_images)

test_images =  test_images[:25]



def read_image(file_path):

    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE

    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)





def prep_data(images):

    count = len(images)

    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)



    for i, image_file in enumerate(images):

        image = read_image(image_file)

        data[i] = image.T

        if i%250 == 0: print('Processed {} of {}'.format(i, count))

    

    return data



train = prep_data(train_images)

test = prep_data(test_images)



print("Train shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))
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
print("Train shape: {}".format(train.shape))

print("Test shape: {}".format(test.shape))
train[0][3]