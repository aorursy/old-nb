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




ss = pd.read_csv('../input/sample_submission.csv')
ss.head()




get_ipython().system('ls -l ../input/ ../input/train')




#%pycat ../input/train/README.md




get_ipython().system('ls -F ../input/train/audio')




import os




labels_ = get_ipython().getoutput('ls -d ../input/train/audio/[a-z]*/')
labels = [os.path.basename(os.path.dirname(p)) for p in labels_]
print(len(labels), labels)




get_ipython().system('head -10 ../input/train/testing_list.txt')




get_ipython().system('head -10 ../input/train/validation_list.txt')




get_ipython().system('ls -l ../input/train/audio/_background_noise_/')




get_ipython().run_line_magic('pycat', '../input/train/audio/_background_noise_/README.md')




from IPython.display import display,Audio,Image,HTML




noise_list = get_ipython().getoutput('ls ../input/train/audio/_background_noise_/*.wav')
noise_list




for w in noise_list[0:3]:
    display(HTML('<h4>{:s}</h4>'.format(os.path.basename(w))))
    display(Audio(w))




import librosa




bed_list = get_ipython().getoutput('ls ../input/train/audio/{labels[0]}/*.wav')
bed_list[-5:]




for w in bed_list[-3:]:
    display(HTML('<h4>{:s}</h4>'.format(os.path.basename(w))))
    display(Audio(w))




bird_list = get_ipython().getoutput('ls ../input/train/audio/{labels[1]}/*.wav')
bird_list[-5:]




yy1, sr1 = librosa.load(bird_list[0], sr=None, dtype=np.float32)
yy1      = librosa.resample(yy1, sr1, 16000)
print(yy1.shape)
display(Audio(yy1,rate=16000))




get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt




import time




def display_wave(yy, sr=16000, filename=None, figsize=None, ylim=None, delay=None):
    if figsize is None: figsize=(6,0.75)
    if ylim is None: ylim=[-0.65,0.65]
    if delay is None: delay=0.2
    display(Audio(yy, rate=sr, filename=filename))
    plt.figure(figsize=figsize)
    plt.ylim(ylim)
    plt.plot(np.arange(len(yy), dtype=np.float32)/sr,yy)
    plt.show()
    time.sleep(delay)




display_wave(yy1)




import tensorflow as tf
print(tf.__version__)






