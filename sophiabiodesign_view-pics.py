#!/usr/bin/env python
# coding: utf-8



import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import cv2
import math
from sklearn import mixture
from sklearn.utils import shuffle
from skimage import measure
from glob import glob
import os
from multiprocessing import Pool, cpu_count
from functools import partial
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

TRAIN_DATA = "../input/train"
ADDITIONAL_DATA_TYPE_1 = "../input/additional_Type_1"

types = ['Type_1','Type_2','Type_3']
type_ids = []
additional_ids = []

for type in enumerate(types):
    type_i_files = glob(os.path.join(TRAIN_DATA, type[1], "*.jpg"))
    type_i_ids = np.array([s[len(TRAIN_DATA)+8:-4] for s in type_i_files])
    type_ids.append(type_i_ids)
    
    additional_files = glob(os.path.join(ADDITIONAL_DATA_TYPE_1, "*.jpg"))
    additional_i_ids = np.array([s[len(ADDITIONAL_DATA_TYPE_1)+8:-4] for s in additional_files])
    additional_ids.append(additional_i_ids)

def get_filename(image_id, image_type):
    """
    Method to get image file path from its id and type   
    """
    if image_type == "Type_1" or         image_type == "Type_2" or         image_type == "Type_3":
        data_path = os.path.join(TRAIN_DATA, image_type)
    elif image_type == "Test":
        data_path = TEST_DATA
    elif image_type == "AType_1" or           image_type == "AType_2" or           image_type == "AType_3":
        data_path = os.path.join(ADDITIONAL_DATA, image_type)
    else:
        raise Exception("Image type '%s' is not recognized" % image_type)

    ext = 'jpg'
    path = os.path.join(data_path, "{}.{}".format(image_id, ext))
    print(path)
    return os.path.join(data_path, "{}.{}".format(image_id, ext))

def get_image_data(image_id, image_type):
    """
    Method to get image data as np.array specifying image id and type
    """
    fname = get_filename(image_id, image_type)
    img = cv2.imread(fname)
    assert img is not None, "Failed to read image : %s, %s" % (image_id, image_type)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)

def show_additional_data(image_id):
    print("Additional Ids:")
    print(additional_ids)
    ext = 'jpg'
    path = os.path.join(ADDITIONAL_DATA_TYPE_1, "{}.{}".format(image_id, ext))
    print(path)
    img = cv2.imread(path)
    assert img is not None, "Failed to read image : %s" % (image_id)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
 
get_image_data(170, type[1])

#3show_additional_data(1790)

