#!/usr/bin/env python
# coding: utf-8



import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




import os
import re
import cv2




import torch
import torchvision




import albumentations as A
from albumentations.pytorch import ToTensor 




from matplotlib.image import imread




from google.colab import drive
drive.mount('/content/drive')

train = pd.read_csv('train.csv')




test = pd.read_csv('sample_submission.csv')




train.head()




def append_ext(fn):
    return fn+".jpg"
train["image_id"]=train["image_id"].apply(append_ext)




train['bbox'].count()




train['xmin'] = None
train['ymin'] = None
train['Width'] = None
train['Height'] = None




def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r

train[['xmin', 'ymin', 'Width', 'Height']] = np.stack(train['bbox'].apply(lambda x: expand_bbox(x)))
train.drop(columns=['bbox'], inplace=True)
train['xmin'] = train['xmin'].astype(np.float)
train['ymin'] = train['ymin'].astype(np.float)
train['Width'] = train['Width'].astype(np.float)
train['Height'] = train['Height'].astype(np.float)




train.head()




train['source'].unique()




sns.countplot(x = 'source', data=train)

# THIS SHOWS THE IMBALANCE IN DATASET




sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

# NO NULL VALUES









train.head()




train.head()




train.drop('width', axis=1, inplace=True)




train.drop('height', axis=1, inplace=True)





data_dir = '/kaggle/input/global-wheat-detection'




os.listdir(data_dir)




train_path = data_dir + '/train/'
test_path = data_dir + '/test/'









a = os.listdir(test_path)[0]





sam_test = test_path + '348a992bb.jpg'




sam_tensor = imread(sam_test)




test_img = test_path+ '51b3e36ab.jpg'




test_img = imread(test_img)




test_img.shape




plt.imshow(test_img)




image_shape = (224,224,3)




from tensorflow.keras.preprocessing.image import ImageDataGenerator




batch_size = 64




test_image_generator = ImageDataGenerator(
    rescale=1./255)




def append_ext(fn):
    return fn+".jpg"
test["image_id"]=test["image_id"].apply(append_ext)




total_samples = 146704




test_generator = test_image_generator.flow_from_dataframe(
    dataframe = test,
    directory=test_path,
     x_col='image_id',
    target_size=(1024,1024),
    shuffle=False,
    batch_size=8,
    class_mode=None
)




from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import LearningRateScheduler
from keras.metrics import *
# v4

ACCURACY_LIST = []
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalMaxPooling2D
from keras.models import Model




METRICS = [
     
      BinaryAccuracy(name='accuracy'),
      Precision(name='precision'),
      Recall(name='recall'),
      AUC(name='auc'),
]

def output_custom_model(prebuilt_model):
    print(f"Processing {prebuilt_model}")
    prebuilt = prebuilt_model(include_top=False,
                            input_shape=(1024,1024,3),
                            weights='imagenet')
    output = prebuilt.output
    output = GlobalMaxPooling2D()(output)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.4)(output)
    output = Dense(7, activation='softmax')(output)

    model = Model(inputs=prebuilt.input, outputs=output)
    model.compile(optimizer='adam', loss=categorical_crossentropy,
              metrics=METRICS)
    return model




def scheduler(epoch):
    if epoch < 5:
        return 0.0001
    else:
        print(f"Learning rate reduced to {0.0001 * np.exp(0.5 * (5 - epoch))}")
        return 0.0001 * np.exp(0.5 * (5 - epoch))
    
custom_callback = LearningRateScheduler(scheduler)




resnet_custom_model = output_custom_model(ResNet50)





test = pd.read_csv('sample_submission.csv')




test.head()






resnet_custom_model.load_weights('wheat_detection.h5')




sam_tensor.shape




sam_tensor = np.expand_dims(sam_tensor,axis=0)




preds = resnet_custom_model.predict(test_generator)




preds.min()









from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.measure import label, regionprops
from PIL import Image, ImageDraw
from ast import literal_eval
from tqdm.notebook import tqdm




IMG_WIDTH = 512
IMG_HEIGHT = 512




SC_FACTOR = int(1024 / IMG_WIDTH)




from tqdm.notebook import tqdm




train1 = pd.read_csv('train.csv')




def make_polygon(coords):
    xm, ym, w, h = coords
    xm, ym, w, h = xm / SC_FACTOR, ym / SC_FACTOR, w / SC_FACTOR, h / SC_FACTOR   # scale values if image was downsized
    return [(xm, ym), (xm, ym + h), (xm + w, ym + h), (xm + w, ym)]

masks = dict() # dictionnary containing all masks

for img_id, gp in tqdm(train1.groupby("image_id")):
    gp['polygons'] = gp['bbox'].apply(eval).apply(lambda x: make_polygon(x))

    img = Image.new('L', (IMG_WIDTH, IMG_HEIGHT), 0)
    for pol in gp['polygons'].values:
        ImageDraw.Draw(img).polygon(pol, outline=1, fill=1)

    mask = np.array(img, dtype=np.uint8)
    masks[img_id] = mask




im = Image.fromarray(masks[list(masks.keys())[7]])
plt.imshow(im)




def show_images(images, num=2):
    
    images_to_show = np.random.choice(images, num)

    for image_id in images_to_show:

        image_path = os.path.join(train_path, image_id + ".jpg")
        image = Image.open(image_path)
  
        # get all bboxes for given image in [xmin, ymin, width, height]
        bboxes = [literal_eval(box) for box in train1[train1['image_id'] == image_id]['bbox']]

        # visualize them
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:    
            draw.rectangle([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]], width=3)

        plt.figure(figsize = (15,15))
        plt.imshow(image)
        plt.show()


unique_images = train1['image_id'].unique()
show_images(unique_images)




THRESH = 0.3
masked_preds = preds > THRESH









def get_params_from_bbox(coords, scaling_factor=1):
    xmin, ymin = coords[1] * scaling_factor, coords[0] * scaling_factor
    w = (coords[3] - coords[1]) * scaling_factor
    h = (coords[2] - coords[0]) * scaling_factor
    
    return xmin, ymin, w, h




bboxes = list()

for j in range(masked_preds.shape[0]):
    label_j = label(masked_preds) 
    props = regionprops(label_j)
    bboxes.append(props)




sample_sub = pd.read_csv('sample_submission.csv')




output = dict()
for i in range(masked_preds.shape[0]):
    bboxes_processed = [get_params_from_bbox(bb.bbox, scaling_factor=SC_FACTOR) for bb in bboxes[i]]
    formated_boxes = ['1.0 ' + ' '.join(map(str, bb_m)) for bb_m in bboxes_processed]
    #if formated_boxes:
    #    formated_boxes = formated_boxes[0] 
    
    output[sample_sub["image_id"][i]] = " ".join(formated_boxes)
    #output[sample_sub["image_id"][i]] = formated_boxes




sample_sub["PredictionString"] = output.values()




sample_sub







sample_sub.to_csv('submission.csv', index=False)











