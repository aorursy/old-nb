# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab as plt

data = pd.read_csv("../input/train.csv")

N = 786240
b = 420

signal = np.zeros(N)
signal_2 = np.zeros(N//b)
daily = np.zeros(10080)
h = np.zeros(10080//60)

for i in data["time"]:
    signal[i] = signal[i] + 1
    signal_2[i//b] = signal_2[i//b] + 1
    daily[(i%10080)] = daily[(i%10080)] + 1
    h[(i%10080)//60] = h[(i%10080)//60] + 1
    
plt.plot(signal)
plt.show()

plt.plot(signal_2)
plt.show()

W    = np.fft.fft(signal)
freq = np.fft.fftfreq(N,1)

plt.plot(1.0/freq[:N//2], abs(W[:N//2]))
plt.xlim(0,20000)
plt.ylim(0,500000)

plt.show()

plt.plot(daily)
plt.show()

plt.plot(h)
plt.show()

W    = np.fft.fft(h)
freq = np.fft.fftfreq(168,1)

plt.plot(1.0/freq[:168//2], abs(W[:168//2]))
plt.xlim(0,168)

plt.show()

one_place = data.loc[data['place_id'] == 9129780742]
    
h = np.zeros(10080//60)
for i in one_place["time"]:
    h[(i%10080//60)] = h[(i%10080//60)] + 1
    
plt.plot(h)
plt.show()


one_place = data.loc[data['place_id'] == 1623394281]
    
h = np.zeros(10080//60)
for i in one_place["time"]:
    h[(i%10080//60)] = h[(i%10080//60)] + 1
    
plt.plot(h)
plt.show() 

one_place = data.loc[data['place_id'] == 1308450003]
    
h = np.zeros(10080//60)
for i in one_place["time"]:
    h[(i%10080//60)] = h[(i%10080//60)] + 1
    
plt.plot(h)
plt.show() 

one_place = data.loc[data['place_id'] == 4371034975]
    
h = np.zeros(10080//60)
for i in one_place["time"]:
    h[(i%10080//60)] = h[(i%10080//60)] + 1
    
plt.plot(h)
plt.show() 

one_place = data.loc[data['place_id'] == 7698408658]
    
h = np.zeros(10080//60)
for i in one_place["time"]:
    h[(i%10080//60)] = h[(i%10080//60)] + 1
    
plt.plot(h)
plt.show() 
