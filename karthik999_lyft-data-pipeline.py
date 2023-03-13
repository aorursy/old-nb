#!/usr/bin/env python
# coding: utf-8



get_ipython().system(' pip install git+https://github.com/lyft/nuscenes-devkit')




from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer, Quaternion, view_points
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

import pandas as pd
import matplotlib.pyplot as plt
import os




train = pd.read_csv("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train.csv")
train.head(2)




train[train["Id"] == "fd5f1c634b59e3b4e0f7a5c48c768a7d334a63221fced363a2ebac41f465830d"]["PredictionString"]




single_object = train.iloc[0]["PredictionString"].split()[:8]
single_object




single_object




int_single_object = [ float(_) for _ in single_object[:6]]
int_single_object.insert(8 , single_object[7] )
int_single_object




from PIL import Image
import matplotlib.pyplot as plt

w=50
h=50
fig=plt.figure(figsize=(20, 20))
columns = 4
rows = 5
for image_index , train_image in enumerate(os.listdir("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images/")):
    img = Image.open("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images/" + train_image )
    fig.add_subplot(rows, columns, image_index+1)
    plt.title(train_image)
    plt.imshow(img)
    if image_index == 19: break
plt.show()




import json

with open("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data/sample_data.json",encoding='utf-8', errors='ignore') as json_data:
     data = json.load(json_data, strict=False)

data[0]       
        




def parse_string_list(single_object):  
   int_single_object = [ float(_) for _ in single_object[:6]]
   int_single_object.insert(8 , single_object[7] )
   return int_single_object




for image_index , train_image in enumerate(os.listdir("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images/")):
    img = Image.open("/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images/" + train_image )
    
    for each_image_data in data:
        filename = each_image_data["filename"]
        filename = filename.split("/")[-1]
        sample_token = each_image_data["sample_token"]
        
        if train_image == filename:
            prediction_string = train[train["Id"] == sample_token]["PredictionString"].values
            prediction_string = prediction_string[0].split()
            for each_object_data in range(0 , len(prediction_string) , 8):
                each_object = prediction_string[each_object_data:each_object_data+8]
                each_object =  parse_string_list(each_object)
                
                plt.imshow(img)
                plt.title(each_object[-1])
                print(each_object)

            break
    break   




get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_images images')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_maps maps')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_lidar lidar')
get_ipython().system('ln -s /kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data data')




lyftdata = LyftDataset(data_path='.', json_path='data/', verbose=True)




token0 = "fd5f1c634b59e3b4e0f7a5c48c768a7d334a63221fced363a2ebac41f465830d" 
my_sample = lyftdata.get('sample', token0)
my_sample




my_sample.keys()




my_annotation = lyftdata.get('sample_annotation', my_sample['anns'][0])
my_annotation




my_box = lyftdata.get_box(my_annotation['token'])
my_box # Box class instance




my_box.center, my_box.wlh # center coordinates + width, length and height




lyftdata.render_annotation(my_annotation['token'], margin=10)




lyftdata.render_sample(token0)

