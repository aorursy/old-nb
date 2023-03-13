#!/usr/bin/env python
# coding: utf-8



import os
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

#The number of sample in every record
dataPoints=800000
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def sine_func(x, a, b, c,d):
    return a * np.sin((b*x)+c)+d




t = pd.read_parquet('../input/train.parquet', engine='pyarrow', columns=['575'])  #8021 8031 8081  
plt.figure(figsize=(24, 10))
plt.plot(t[:],label='Raw series')
plt.legend(loc='best')
plt.show()




y_data = t[0:dataPoints].values.reshape(dataPoints)
x_data = np.linspace(0,1, num=dataPoints).T.reshape(dataPoints)
y_data_smooth = smooth(y_data, 10000)
plt.figure(figsize=(24, 10))
plt.plot( y_data,'r+')
plt.plot(y_data_smooth,"b-",label='Smoothed Data')
plt.legend(loc='best')
plt.show()




peaks, _ = find_peaks(y_data_smooth,threshold =(0.00001,0.0001),width = 100)
plt.plot(y_data_smooth)
plt.plot(peaks, y_data_smooth[peaks], "x")
plt.plot(np.zeros_like(y_data_smooth), "--", color="gray",label='Peaks')
plt.legend(loc='best')
plt.show()




# The more than two peaks are found. We cluster this peak into two clusters and take average 
# for finding their position on x-axis. Kmeans cluster with 2 numbers of clusters returns the cluster 
# index for every peak. We check which is peak is positive by comparing them and take average of 
# peak positions for it. From this location of positive peak we calculated the phase with below reasoning.

# The entire signel is capturing a full wave i.e 360 degress or 2 pi in 800000 time slices. 
# So the every displacement counts for 2*pi/800000. 
# The positive peack will appear at pi/2 for zero phase. and hence
# phase = (((800000- posivePeackLoc)+200000)% 800000)/800000*np.pi/2         ---phase formula

# The sine wave has equation,
# y(t) = A\sin(2 \pi f t + \varphi) = A\sin(\omega t + \varphi)
# where:

# A = the amplitude, the peak deviation of the function from zero.
# f = the ordinary frequency, the number of oscillations (cycles) that occur each second of time.
# ω = 2πf, the angular frequency, the rate of change of the function argument in 
#         units of radians per second 
# {\displaystyle \varphi } \varphi  = the phase, specifies (in radians) where in its cycle 
#         the oscillation is at t = 0.    When {\displaystyle \varphi } \varphi  is non-zero, 
#     the entire waveform appears to be shifted in time by the amount {\displaystyle \varphi } 
#     \varphi /ω seconds. A negative value represents a delay, and a positive value represents an advance.

# We also add a displacement parameter so the fitted curve can move horizontally.

# The intial guasses are below,
# A = 20 visual inspection  and problem description.
# f = 1/800000. There are 800000 samples per full cycle.
# ω = 2πf = 4*np.pi/800000
# phase = (((800000- posivePeackLoc)+200000)% 800000)/800000*np.pi/2         ---phase formula 

# The fitted wave is plotted in blue over red raw signal. Good fit!




km = KMeans(n_clusters=2)
km.fit(peaks.reshape(-1,1))
clu = km.predict(peaks.reshape(-1,1))

if ( np.mean(y_data_smooth[peaks[clu==1]]) > np.mean(y_data_smooth[peaks[clu==0]])) :
        posivePeackLoc = np.mean(peaks[clu==1])
else:
        posivePeackLoc = np.mean(peaks[clu==0])

y_data = t[0:dataPoints].values.reshape(dataPoints)
x_data = np.linspace(0,1, num=dataPoints).T.reshape(dataPoints)
y_data[y_data>45]=45
y_data[y_data<-45]=-45
phase = (((800000- posivePeackLoc)+200000)% 800000)/800000*np.pi/2
params, params_covariance = optimize.curve_fit(sine_func, x_data, y_data,
                                               p0=[20, 4*np.pi/800000, phase,5])

plt.figure(figsize=(24, 10))
plt.plot( y_data,'r+')
plt.plot(x_data*dataPoints, sine_func(x_data, params[0], params[1],params[2],params[3]),
         label='Fitted Sine Wave')
plt.legend(loc='best')
plt.show()




residual=y_data-sine_func(x_data, params[0], params[1],params[2],params[3])
plt.figure(figsize=(24, 10))
plt.plot(x_data*dataPoints, y_data-sine_func(x_data, params[0], params[1],params[2],params[3]),
         label='Residual')
plt.legend(loc='best')
plt.show()

