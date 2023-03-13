#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np




from glob import glob

def make_base_df():
    base_path = '../input/train'
    image_paths = []
    for type_base_path in sorted(glob(base_path +'/*')):
        image_paths = image_paths + glob(type_base_path + '/*')
    df = pd.DataFrame({'path':image_paths})
    df['type'] = df.path.map(lambda x: x.split('/')[-2])
    df['filetype'] = df.path.map(lambda x: x.split('.')[-1])
    df['num_id'] = df.path.map(lambda x:x.split('/')[-1].split('.')[0])
    return df




import cv2

df = make_base_df()
path = df.path[1]
img = cv2.imread(path)
print(img)




import cv2

def get_grayscale_img(path, rescale_dim):
    img = cv2.imread(path)
    rescaled = cv2.resize(img, (rescale_dim, rescale_dim), cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rescaled, cv2.COLOR_RGB2GRAY).astype('float')
    return gray

def save_img(img, num_id, desc):
    path = desc+num_id+".jpg"
    cv2.imwrite(path,img)
    return path
    
def save_grayscale(row, desc):
    path = row.path
    num_id = row.num_id
    gray = get_grayscale_img(path, 100)
    return save_img(gray, num_id, desc)




df = make_base_df()
small_df = df[:100]
desc = 'gray_'
for index, row in small_df.iterrows():
    small_df.loc[index,'gray_path'] = save_grayscale(row, desc)
    
small_df.head()




import matplotlib.pyplot as plt

df = make_base_df()
small_df = df[:100]
path = df.path[1]
grey = get_grayscale_img(path, 100)
cv2.imwrite("grey.jpg", grey)
plt.imshow(plt.imread("grey.jpg"))




plt.imshow(plt.imread(path))




plt.imshow(plt.imread("grey.jpg"))




from sklearn.ensemble import RandomForestClassifier as RFC

forest = RFC(n_jobs=2,n_estimators=50)

