import pandas as pd

import numpy as np

from PIL import Image, ImageDraw, ImageFilter

from PIL import ImageChops, ImageStat

import cv2, glob, scipy

from scipy import ndimage



train = sorted(glob.glob('../input/train/*.jpg'))

masks = sorted(glob.glob('../input/train_masks/*.gif'))

test = sorted(glob.glob('../input/test/*.jpg'))

print(len(train), len(masks), len(test))



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import plotly.plotly as py


plt.rcParams['figure.figsize'] = (10.0, 10.0)



im = Image.open(train[1]).convert('LA');

#plt.imshow(im);

'''

w, h = im.size  

colors = im.getcolors(w*h)



def hexencode(rgb):

    r=rgb[0]

    g=rgb[1]

    b=rgb[2]

    return '#%02x%02x%02x' % (r,g,b)



for idx, c in enumerate(colors):

    plt.bar(idx, c[0], color=hexencode(c[1]))



plt.show()*/

'''

num_im = np.array(im);

plt.hist(num_im.ravel(), bins=256, range=(0, 255))

plt.title('Histogram for gray scale picture')

plt.show()



# Histogram Equalization





hist,bins = np.histogram(num_im.flatten(),256,[0,256])



cdf = hist.cumsum()

cdf_normalized = cdf * hist.max()/ cdf.max()



plt.plot(cdf_normalized, color = 'b')

plt.hist(num_im.flatten(),256,[0,256], color = 'r')

plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'upper left')

plt.show()



cdf_m = np.ma.masked_equal(cdf,0)

cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())

cdf = np.ma.filled(cdf_m,0).astype('uint8')



img2 = cdf[num_im]



plt.hist(img2.flatten(),256,[0,256], color = 'r')



plt.figure()

plt.imshow(im) 

plt.show()

histEqualizedImage = Image.fromarray(img2)

plt.figure()

plt.imshow(histEqualizedImage) 

plt.show() 



from collections import Counter

  

def Thresholding_Otsu(img):

    nbins = 256  # or np.max(img)-np.min(img) for images with non-regular pixel values

    pixel_counts = Counter(img.ravel())

    counts = np.array([0 for x in range(nbins)])

    for c in sorted(pixel_counts):

        counts[c] = pixel_counts[c]

    p = counts / sum(counts)

    sigma_b = np.zeros((nbins, 1))

    for t in range(nbins):

        q_L = sum(p[:t])

        q_H = sum(p[t:])

        if q_L == 0 or q_H == 0:

            continue



        miu_L = sum(np.dot(p[:t], np.transpose(np.matrix([i for i in

                    range(t)])))) / q_L

        miu_H = sum(np.dot(p[t:], np.transpose(np.matrix([i for i in

                    range(t, nbins)])))) / q_H

        sigma_b[t] = q_L * q_H * (miu_L - miu_H) ** 2



    return np.argmax(sigma_b)



otsuThresholdedImageArray = Thresholding_Otsu(img2);

#plt.plot(otsuThresholdedImageArray)

print(otsuThresholdedImageArray)

#otsuThresholdedImage = Image.fromarray(otsuThresholdedImageArray)

#plt.figure()

#plt.imshow(otsuThresholdedImage) 

#plt.show() 

# Autogenerated with SMOP version 

# main.py EMDSegmentation.m



from __future__ import division

try:

    from runtime import *

except ImportError:

    from smop.runtime import *



clc

close(char('all'))

clear(char('all'))

Adj=1

Name,Path=uigetfile(char('*.bmp'),char('Browse BMP Image'),nargout=2)

FileName=strcat(Path,Name)

DataArray,_map=imread(FileName,nargout=2)

DataArrayBackup=copy_(DataArray)

figure

imshow(DataArray,_map)

title(char('Original Image'))

ImH,ImW=size(DataArray,nargout=2)

DataArray=double(DataArray)

StartTime=copy_(cputime)

HistogramArray=zeros(1,256)

HistogramArray=uint32(HistogramArray)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        HistogramArray[1,DataArray(i + Adj,j + Adj) + Adj]=HistogramArray(1,DataArray(i + Adj,j + Adj) + Adj) + 1

figure

bar(HistogramArray)

title(char('Histogram of Original image'))

BlockSize=15

OptimumThreshold=128

PMin=0.0

P2=20 / 100.0

PMin=P2 * ImH * ImW

SumRange1=0.0

for i in arange_(0,OptimumThreshold).reshape(-1):

    SumRange1=SumRange1 + HistogramArray(1,i + Adj)

SumRange2=0.0

for i in arange_(255,OptimumThreshold + 1,- 1).reshape(-1):

    SumRange2=SumRange2 + HistogramArray(1,i + Adj)

SeedXj=0

SeedXr=0

if (SumRange1 < PMin or SumRange2 < PMin):

    DataArray=histeq(uint8(DataArray))

    figure

    imshow(uint8(DataArray))

    title(char('Hist Equalised image'))

    for i in arange_(0,256 - 1).reshape(-1):

        HistogramArray[1,i + Adj]=0

    for i in arange_(0,ImH - 1).reshape(-1):

        for j in arange_(0,ImW - 1).reshape(-1):

            HistogramArray[1,DataArray(i + Adj,j + Adj) + Adj]=HistogramArray(1,DataArray(i + Adj,j + Adj) + Adj) + 1

    SeedXj=0

    SumRange1=0.0

    for i in arange_(0,OptimumThreshold).reshape(-1):

        SumRange1=SumRange1 + HistogramArray(1,i + Adj)

        if (SumRange1 > (PMin)):

            SeedXj=copy_(i)

            break

    SumRange2=0.0

    SeedXr=0

    for i in arange_(255,OptimumThreshold + 1,- 1).reshape(-1):

        SumRange2=SumRange2 + HistogramArray(1,i + Adj)

        if (SumRange2 > (PMin)):

            SeedXr=copy_(i)

            break

else:

    DataArray=copy_(DataArray)

    figure

    imshow(uint8(DataArray))

    title(char('Non Equalised image'))

    for i in arange_(0,256 - 1).reshape(-1):

        HistogramArray[1,i + Adj]=0

    for i in arange_(0,ImH - 1).reshape(-1):

        for j in arange_(0,ImW - 1).reshape(-1):

            HistogramArray[1,DataArray(i + Adj,j + Adj) + Adj]=HistogramArray(1,DataArray(i + Adj,j + Adj) + Adj) + 1

    SeedXj=0

    SumRange1=0.0

    for i in arange_(0,OptimumThreshold).reshape(-1):

        SumRange1=SumRange1 + HistogramArray(1,i + Adj)

        if (SumRange1 > (PMin)):

            SeedXj=copy_(i)

            break

    SeedXr=0

    SumRange2=0.0

    for i in arange_(255,OptimumThreshold + 1,- 1).reshape(-1):

        SumRange2=SumRange2 + HistogramArray(1,i + Adj)

        if (SumRange2 > (PMin)):

            SeedXr=copy_(i)

            break

DataArrayBackup=copy_(DataArray)

DataArrayBackup=double(DataArrayBackup)

BackupSeedXr=copy_(SeedXr)

BackupSeedXj=copy_(SeedXj)

SeedXMin=0

for i in arange_(0,255).reshape(-1):

    if (HistogramArray(1,i + Adj) > 0):

        SeedXMin=copy_(i)

        break

SeedXMax=0

for i in arange_(255,0,- 1).reshape(-1):

    if (HistogramArray(1,i + Adj) > 0):

        SeedXMax=copy_(i)

        break

a=0

c=0

b=0

DataArray=double(DataArray)

p=copy_(SeedXMin)

q=copy_(SeedXMax)

DenominatorSum=0.0

NumeratorSum=0.0

for m in arange_(p,q).reshape(-1):

    NumeratorSum=NumeratorSum + m * HistogramArray(1,m + Adj)

    DenominatorSum=DenominatorSum + HistogramArray(1,m + Adj)

b=NumeratorSum / DenominatorSum

MaxValue=0.0

if (_abs(b - SeedXMin) < _abs(SeedXMax - b)):

    MaxValue=_abs(b - SeedXMin)

else:

    MaxValue=_abs(SeedXMax - b)

c=b + MaxValue

a=(2 * b) - c

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (DataArray(i + Adj,j + Adj) >= SeedXj):

            DataArray[i + Adj,j + Adj]=DataArray(i + Adj,j + Adj)

        else:

            DataArray[i + Adj,j + Adj]=0

MewA=zeros(ImH,ImW)

MewA=double(MewA)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (DataArray(i + Adj,j + Adj) < a):

            MewA[i + Adj,j + Adj]=0

        else:

            if (DataArray(i + Adj,j + Adj) >= a and DataArray(i + Adj,j + Adj) < b):

                MewA[i + Adj,j + Adj]=(((DataArray(i + Adj,j + Adj) - double(a)) / (double(c) - double(a))) ** 2) * 2.0

            else:

                if (DataArray(i + Adj,j + Adj) >= b and DataArray(i + Adj,j + Adj) < c):

                    MewA[i + Adj,j + Adj]=1.0 - (((DataArray(i + Adj,j + Adj) - double(a)) / (double(c) - double(a))) ** 2) * 2.0

                else:

                    MewA[i + Adj,j + Adj]=1.0

figure

imshow(uint8(MewA * 255))

title(char('MewA In Bright Object Image'))

MewA_star=zeros(ImH,ImW)

MewA_star=double(MewA_star)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (MewA(i + Adj,j + Adj) < 0.5):

            MewA_star[i + Adj,j + Adj]=0.0

        else:

            MewA_star[i + Adj,j + Adj]=1.0

MewAStarBrightObjectArray=zeros(ImH,ImW)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        MewAStarBrightObjectArray[i + Adj,j + Adj]=MewA_star(i + Adj,j + Adj)

figure

imshow(uint8(MewA_star * 255))

title(char('MewA_star In Bright Object Image'))

SumValue=0.0

SumValue=double(SumValue)

Chy_BrighterObject=0.0

Chy_BrighterObject=double(Chy_BrighterObject)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        SumValue=SumValue + (MewA(i + Adj,j + Adj) - MewA_star(i + Adj,j + Adj)) ** 2

Chy_BrighterObject=(sqrt(SumValue) * 2.0) / sqrt(ImH * ImW)

disp(char('Chy_BrighterObject is ='))

disp(Chy_BrighterObject)

Chy_BrighterObject_ForSingleElementArray=zeros(ImH,ImW)

Chy_BrighterObject_ForSingleElementArray=double(Chy_BrighterObject_ForSingleElementArray)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        Chy_BrighterObject_ForSingleElementArray[i + Adj,j + Adj]=2 * (sqrt((MewA(i + Adj,j + Adj) - MewA_star(i + Adj,j + Adj)) ** 2)) / sqrt(1)

figure

imshow(uint8((Chy_BrighterObject_ForSingleElementArray / _max(_max(Chy_BrighterObject_ForSingleElementArray))) * 255))

title(char('Chy_BrighterObject_ForSingleElementArray'))

figure

bar(imhist(uint8((Chy_BrighterObject_ForSingleElementArray / _max(_max(Chy_BrighterObject_ForSingleElementArray))) * 255)))

title(char('Histogram for Chy_BrighterObject'))

SeedXMin=0

for i in arange_(0,255).reshape(-1):

    if (HistogramArray(1,i + Adj) > 0):

        SeedXMin=copy_(i)

        break

SeedXMax=0

for i in arange_(255,0,- 1).reshape(-1):

    if (HistogramArray(1,i + Adj) > 0):

        SeedXMax=copy_(i)

        break

a=0

c=0

b=0

DataArray=double(DataArray)

p=copy_(SeedXMin)

q=copy_(SeedXMax)

DenominatorSum=0.0

NumeratorSum=0.0

for m in arange_(p,q).reshape(-1):

    NumeratorSum=NumeratorSum + m * HistogramArray(1,m + Adj)

    DenominatorSum=DenominatorSum + HistogramArray(1,m + Adj)

b=NumeratorSum / DenominatorSum

MaxValue=0.0

if (_abs(b - SeedXMin) < _abs(SeedXMax - b)):

    MaxValue=_abs(b - SeedXMin)

else:

    MaxValue=_abs(SeedXMax - b)

c=b + MaxValue

a=(2 * b) - c

DataArray=copy_(DataArrayBackup)

DataArray=double(DataArray)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (DataArray(i + Adj,j + Adj) < SeedXr):

            DataArray[i + Adj,j + Adj]=DataArray(i + Adj,j + Adj)

        else:

            DataArray[i + Adj,j + Adj]=255

MewA=zeros(ImH,ImW)

MewA=double(MewA)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (DataArray(i + Adj,j + Adj) < a):

            MewA[i + Adj,j + Adj]=1

        else:

            if (DataArray(i + Adj,j + Adj) >= a and DataArray(i + Adj,j + Adj) < b):

                MewA[i + Adj,j + Adj]=1.0 - (((DataArray(i + Adj,j + Adj) - double(a)) / (double(c) - double(a))) ** 2) * 2.0

            else:

                if (DataArray(i + Adj,j + Adj) >= b and DataArray(i + Adj,j + Adj) < c):

                    MewA[i + Adj,j + Adj]=(((DataArray(i + Adj,j + Adj) - double(a)) / (double(c) - double(a))) ** 2) * 2.0

                else:

                    MewA[i + Adj,j + Adj]=0.0

figure

imshow(uint8(MewA * 255))

title(char('MewA image In Dark Object Image'))

MewA_star=zeros(ImH,ImW)

MewA_star=double(MewA_star)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (MewA(i + Adj,j + Adj) < 0.5):

            MewA_star[i + Adj,j + Adj]=1.0

        else:

            MewA_star[i + Adj,j + Adj]=0.0

MewAStarDarkObjectArray=zeros(ImH,ImW)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        MewAStarDarkObjectArray[i + Adj,j + Adj]=MewA_star(i + Adj,j + Adj)

figure

imshow(uint8(MewA_star * 255))

title(char('MewA_star In Dark Object Image'))

SumValue=0.0

SumValue=double(SumValue)

Chy_DarkerObject=0.0

Chy_DarkerObject=double(Chy_DarkerObject)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        SumValue=SumValue + (MewA(i + Adj,j + Adj) - MewA_star(i + Adj,j + Adj)) ** 2

Chy_DarkerObject=(sqrt(SumValue) * 2.0) / sqrt(ImH * ImW)

disp(char('Chy_DarkerObject ='))

disp(Chy_DarkerObject)

Chy_DarkerObject_ForSingleElementArray=zeros(ImH,ImW)

Chy_DarkerObject_ForSingleElementArray=double(Chy_DarkerObject_ForSingleElementArray)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        Chy_DarkerObject_ForSingleElementArray[i + Adj,j + Adj]=2 * (sqrt((MewA(i + Adj,j + Adj) - MewA_star(i + Adj,j + Adj)) ** 2)) / sqrt(1)

figure

imshow(uint8((Chy_DarkerObject_ForSingleElementArray / _max(_max(Chy_DarkerObject_ForSingleElementArray))) * 255))

title(char('Chy_DarkerObject_ForSingleElementArray'))

figure

bar(imhist(uint8((Chy_DarkerObject_ForSingleElementArray / _max(_max(Chy_DarkerObject_ForSingleElementArray))) * 255)))

title(char('Histogram for Chy_DarkerObject'))

AlphaValue=0.0

AlphaValue=double(AlphaValue)

AlphaValue=(Chy_BrighterObject / (Chy_DarkerObject))

AlphaValue=1

SegmentedImage=zeros(ImH,ImW)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (DataArray(i + Adj,j + Adj) <= SeedXj):

            SegmentedImage[i + Adj,j + Adj]=0

        else:

            if (DataArray(i + Adj,j + Adj) >= SeedXr):

                SegmentedImage[i + Adj,j + Adj]=1

            else:

                if (Chy_BrighterObject_ForSingleElementArray(i + Adj,j + Adj) < (AlphaValue * Chy_DarkerObject_ForSingleElementArray(i + Adj,j + Adj))):

                    SegmentedImage[i + Adj,j + Adj]=1

                else:

                    SegmentedImage[i + Adj,j + Adj]=0

figure

imshow(SegmentedImage)

title(char('SegmentedImage'))

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (MewAStarBrightObjectArray(i + Adj,j + Adj) == 1):

            SegmentedImage[i + Adj,j + Adj]=1

for i in arange_(1,ImH - 2).reshape(-1):

    for j in arange_(1,ImW - 2).reshape(-1):

        if (MewAStarDarkObjectArray(i + Adj,j + Adj) == 1 and MewAStarDarkObjectArray(i - 1 + Adj,j - 1 + Adj) == 0 and MewAStarDarkObjectArray(i - 1 + Adj,j + 0 + Adj) == 0 and MewAStarDarkObjectArray(i - 1 + Adj,j + 1 + Adj) == 0 and MewAStarDarkObjectArray(i + 0 + Adj,j - 1 + Adj) == 0 and MewAStarDarkObjectArray(i + 0 + Adj,j + 1 + Adj) == 0 and MewAStarDarkObjectArray(i + 1 + Adj,j - 1 + Adj) == 0 and MewAStarDarkObjectArray(i + 1 + Adj,j + 0 + Adj) == 0 and MewAStarDarkObjectArray(i + 1 + Adj,j + 1 + Adj) == 0):

            SegmentedImage[i + Adj,j + Adj]=1

figure

imshow(SegmentedImage * 255)

title(char('Filtered SegmentedImage'))

DataArray=copy_(DataArrayBackup)

MaxDiscrepancySementedImage=copy_(SegmentedImage)

BlockSize=15

TempForegroundCount=0.0

TempForegroundCount=double(TempForegroundCount)

TempBackgroundCount=0.0

TempBackgroundCount=double(TempBackgroundCount)

for i in arange_(0,ImH - 1 - BlockSize,BlockSize).reshape(-1):

    for j in arange_(0,ImW - 1 - BlockSize,BlockSize).reshape(-1):

        if (MaxDiscrepancySementedImage(i + Adj,j + Adj) == 1):

            TempForegroundCount=TempForegroundCount + 1

        else:

            TempBackgroundCount=TempBackgroundCount + 1

if (TempForegroundCount > TempBackgroundCount):

    for i in arange_(0,ImH - 1).reshape(-1):

        for j in arange_(0,ImW - 1).reshape(-1):

            MaxDiscrepancySementedImage[i + Adj,j + Adj]=(1 - MaxDiscrepancySementedImage(i + Adj,j + Adj))

    for i in arange_(0,BlockSize - 1).reshape(-1):

        for j in arange_(0,ImW - 1).reshape(-1):

            MaxDiscrepancySementedImage[i + Adj,j + Adj]=0

    for i in arange_(ImH - 1 - BlockSize - 1,ImH - 1).reshape(-1):

        for j in arange_(0,ImW - 1).reshape(-1):

            MaxDiscrepancySementedImage[i + Adj,j + Adj]=0

    for i in arange_(0,ImH - 1).reshape(-1):

        for j in arange_(0,BlockSize - 1).reshape(-1):

            MaxDiscrepancySementedImage[i + Adj,j + Adj]=0

    for i in arange_(0,ImH - 1).reshape(-1):

        for j in arange_(ImW - 1,ImW - 1 - BlockSize - 1,- 1).reshape(-1):

            MaxDiscrepancySementedImage[i + Adj,j + Adj]=0

EdgeImage=edge(MaxDiscrepancySementedImage,char('canny'))

figure

imshow(EdgeImage)

title(char('EdgeImage in Initial trial stage'))

MarkedImage=copy_(DataArray)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (EdgeImage(i + Adj,j + Adj) == 1):

            MarkedImage[i + Adj,j + Adj]=255

figure

imshow(uint8(MarkedImage))

title(char('MarkedImage in initial trial stage'))

BinaryMarkedImageBackup=zeros(ImH,ImW)

SegmentedImage=zeros(ImH,ImW)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        SegmentedImage[i + Adj,j + Adj]=MaxDiscrepancySementedImage(i + Adj,j + Adj)

BlockSize1=copy_(BlockSize)

for itt in arange_(0,1).reshape(-1):

    InsideCumSum=zeros(1,256)

    InsideCumSum=double(InsideCumSum)

    OutsideCumSum=zeros(1,256)

    OutsideCumSum=double(OutsideCumSum)

    InsideHistogram=zeros(1,256)

    InsideHistogram=double(InsideHistogram)

    OutsideHistogram=zeros(1,256)

    OutsideHistogram=double(OutsideHistogram)

    InsideCount=0

    InsideCount=double(InsideCount)

    OutsideCount=0

    OutsideCount=double(OutsideCount)

    for i in arange_(0,ImH - 1).reshape(-1):

        for j in arange_(0,ImW - 1).reshape(-1):

            if (SegmentedImage(i + Adj,j + Adj) == 1):

                InsideHistogram[1,DataArray(i + Adj,j + Adj) + Adj]=InsideHistogram(1,DataArray(i + Adj,j + Adj) + Adj) + 1

                InsideCount=InsideCount + 1

            else:

                OutsideHistogram[1,DataArray(i + Adj,j + Adj) + Adj]=OutsideHistogram(1,DataArray(i + Adj,j + Adj) + Adj) + 1

                OutsideCount=OutsideCount + 1

    InsideCumSum=cumsum(InsideHistogram)

    OutsideCumSum=cumsum(OutsideHistogram)

    InsideCumSum=(InsideCumSum / InsideCount)

    OutsideCumSum=(OutsideCumSum / OutsideCount)

    BlockSize=9

    BlockHist=zeros(1,256)

    BlockHist=double(BlockHist)

    FinalMarkedImage=zeros(ImH,ImW)

    FinalMarkedImage[arange_(),arange_()]=0

    ForegroundDist=0.0

    ForegroundDist=double(ForegroundDist)

    BackgroundDist=0.0

    BackgroundDist=double(BackgroundDist)

    for i in arange_(0 + fix(BlockSize / 2),ImH - 1 - fix(BlockSize / 2)).reshape(-1):

        for j in arange_(0 + fix(BlockSize / 2),ImW - 1 - fix(BlockSize / 2)).reshape(-1):

            BlockHist[arange_(),arange_()]=0.0

            for k in arange_(- fix(BlockSize / 2),fix(BlockSize / 2)).reshape(-1):

                for L in arange_(- fix(BlockSize / 2),fix(BlockSize / 2)).reshape(-1):

                    BlockHist[1,DataArray(i + k + Adj,j + L + Adj) + Adj]=BlockHist(1,DataArray(i + k + Adj,j + L + Adj) + Adj) + 1

            BlockCumSum=cumsum(BlockHist) / (BlockSize * BlockSize)

            ForegroundDist=0.0

            BackgroundDist=0.0

            for ii in arange_(0,256 - 1).reshape(-1):

                ForegroundDist=ForegroundDist + _abs(BlockCumSum(1,ii + Adj) - InsideCumSum(1,ii + Adj))

                BackgroundDist=BackgroundDist + _abs(BlockCumSum(1,ii + Adj) - OutsideCumSum(1,ii + Adj))

            if (ForegroundDist < BackgroundDist):

                FinalMarkedImage[i + Adj,j + Adj]=1

            else:

                FinalMarkedImage[i + Adj,j + Adj]=0

    figure

    imshow(uint8(FinalMarkedImage * 255))

    title(char('Partial  Refinement Image'))

    BinaryMarkedImageBackup=copy_(FinalMarkedImage)

    for i in arange_(0,ImH - 1).reshape(-1):

        for j in arange_(0,ImW - 1).reshape(-1):

            SegmentedImage[i + Adj,j + Adj]=FinalMarkedImage(i + Adj,j + Adj)

    EdgeImage=edge(FinalMarkedImage,char('canny'))

    figure

    imshow(uint8(EdgeImage * 255))

    title(char('EdgeImage refinement process'))

    MarkedImage=copy_(DataArray)

    for i in arange_(0,ImH - 1).reshape(-1):

        for j in arange_(0,ImW - 1).reshape(-1):

            if (EdgeImage(i + Adj,j + Adj) == 1):

                MarkedImage[i + Adj,j + Adj]=255

    figure

    imshow(uint8(MarkedImage))

    title(char('Refinement process MarkedImage'))

figure

imshow(uint8(MarkedImage))

title(char('Final Segmented Image'))

NoOfPixelsInBackgroundObject=0.0

OutputImage1=zeros(ImH,ImW)

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (BinaryMarkedImageBackup(i + Adj,j + Adj) == 0):

            OutputImage1[i + Adj,j + Adj]=DataArrayBackup(i + Adj,j + Adj)

            NoOfPixelsInBackgroundObject=NoOfPixelsInBackgroundObject + 1

figure

imshow(uint8(OutputImage1))

title(char('Background Object'))

disp(char('NoOfPixelsInBackgroundObject='))

disp(NoOfPixelsInBackgroundObject)

OutputImage2=zeros(ImH,ImW)

NoOfPixelsInForegroundObject=0.0

for i in arange_(0,ImH - 1).reshape(-1):

    for j in arange_(0,ImW - 1).reshape(-1):

        if (BinaryMarkedImageBackup(i + Adj,j + Adj) == 1):

            OutputImage2[i + Adj,j + Adj]=DataArrayBackup(i + Adj,j + Adj)

            NoOfPixelsInForegroundObject=NoOfPixelsInForegroundObject + 1

figure

imshow(uint8(OutputImage2))

title(char('Foreground Object'))

disp(char('NoOfPixelsInForegroundObject='))

disp(NoOfPixelsInForegroundObject)

TimeTaken=cputime() - StartTime

disp(char('TimeTaken for EMD method'))

disp(TimeTaken)
