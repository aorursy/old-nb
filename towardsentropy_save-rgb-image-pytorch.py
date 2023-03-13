#!/usr/bin/env python
# coding: utf-8



get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')




import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import torch
import PIL
from pathlib import Path




path = Path('../input')




def pil2tensor(image,dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False) )




def _load_dataset(base_path, dataset, include_controls=True):
    df =  pd.read_csv(os.path.join(base_path, dataset + '.csv'))
    if include_controls:
        controls = pd.read_csv(
            os.path.join(base_path, dataset + '_controls.csv'))
        df['well_type'] = 'treatment'
        df = pd.concat([controls, df], sort=True)
    df['cell_type'] = df.experiment.str.split("-").apply(lambda a: a[0])
    df['dataset'] = dataset
    dfs = []
    for site in (1, 2):
        df = df.copy()
        df['site'] = site
        dfs.append(df)
    res = pd.concat(dfs).sort_values(
        by=['id_code', 'site']).set_index('id_code')
    return res




def combine_metadata(base_path=path,
                     include_controls=True):
    df = pd.concat(
        [
            _load_dataset(
                base_path, dataset, include_controls=include_controls)
            for dataset in ['test', 'train']
        ],
        sort=True)
    return df




md = combine_metadata()




md.head()




def image_path(dataset, experiment, plate,
               address, site, channel, base_path=path):

    return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                        "{}_s{}_w{}.png".format(address, site, channel))




def open_6_channel(dataset, experiment, plate, address, site, base_path=path):
    return torch.cat([pil2tensor(PIL.Image.open(image_path(dataset, experiment, plate, address, site, i)), np.float32) for i 
            in range(1,7)])




DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)




RGB_MAP = {
    1: {
        'rgb': np.array([19, 0, 249]),
        'range': [0, 51]
    },
    2: {
        'rgb': np.array([42, 255, 31]),
        'range': [0, 107]
    },
    3: {
        'rgb': np.array([255, 0, 25]),
        'range': [0, 64]
    },
    4: {
        'rgb': np.array([45, 255, 252]),
        'range': [0, 191]
    },
    5: {
        'rgb': np.array([250, 0, 253]),
        'range': [0, 89]
    },
    6: {
        'rgb': np.array([254, 255, 40]),
        'range': [0, 191]
    }
}




def convert_tensor_to_rgb(t, channels=DEFAULT_CHANNELS, vmax=255, rgb_map=RGB_MAP):

    t = t.permute(1,2,0).numpy()
    colored_channels = []
    for i, channel in enumerate(channels):
        x = (t[:, :, i] / vmax) /             ((rgb_map[channel]['range'][1] - rgb_map[channel]['range'][0]) / 255) +             rgb_map[channel]['range'][0] / 255
        x = np.where(x > 1., 1., x)
        x_rgb = np.array(
            np.outer(x, rgb_map[channel]['rgb']).reshape(512, 512, 3),
            dtype=int)
        colored_channels.append(x_rgb)
    im = np.array(np.array(colored_channels).sum(axis=0), dtype=int)
    im = np.where(im > 255, 255, im)
    return im




def save_path(dataset, experiment, plate,
               address, site, base_path=path):

    return os.path.join(base_path, dataset, experiment, "Plate{}".format(plate),
                        "{}_s{}.png".format(address, site))




def save_rgb(dataset, experiment, plate, address, site):
    im_6 = open_6_channel(dataset, experiment, plate, address, site)
    im_rgb = convert_tensor_to_rgb(im_6)
    dest = dataset + '_rgb'
    save_file = save_path(dest, experiment, plate, address, site)
    PIL.Image.fromarray(im_rgb.astype('uint8')).save(save_file)




os.mkdir(path/'train_rgb')
for folder in os.listdir(path/'train'):
    os.mkdir(path/'train_rgb'/folder)
    os.mkdir(path/'train_rgb'/folder/'Plate1')
    os.mkdir(path/'train_rgb'/folder/'Plate2')
    os.mkdir(path/'train_rgb'/folder/'Plate3')
    os.mkdir(path/'train_rgb'/folder/'Plate4')
    
os.mkdir(path/'test_rgb')
for folder in os.listdir(path/'test'):
    os.mkdir(path/'test_rgb'/folder)
    os.mkdir(path/'test_rgb'/folder/'Plate1')
    os.mkdir(path/'test_rgb'/folder/'Plate2')
    os.mkdir(path/'test_rgb'/folder/'Plate3')
    os.mkdir(path/'test_rgb'/folder/'Plate4')




_ = md.apply(lambda row: 
            save_rgb(row['dataset'], row['experiment'], row['plate'], row['well'], row['site']), axis=1)

