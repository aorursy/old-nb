import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns




pal = sns.color_palette()



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



for f in os.listdir('../input/'):

    path = os.path.join('../input/', f)

    if not os.path.isdir(path):

        size = os.path.getsize(path) / (1024*1024)

        size = round(size, 2)

        print (f.ljust(30) + str(size) + 'MB')

    else:

        sizes = [os.path.getsize(os.path.join(path, subfile)) / (1024*1024) for subfile in os.listdir(path)]

        print (f.ljust(30) + ('{}MB ({} files)').format(str(round(sum(sizes), 2)), len(sizes)))
df_train = pd.read_csv('../input/train.csv')

df_train.head()
labels = df_train['tags'].apply(lambda x: x.split(' '))
labels = df_train['tags'].apply(lambda x: x.split(' '))

from collections import Counter, defaultdict

counts = defaultdict(int)

for l in labels:

    for l2 in l:

        counts[l2] += 1



data=[go.Bar(x=list(counts.keys()), y=list(counts.values()))]

layout=dict(height=800, width=800, title='Distribution of training labels')

fig=dict(data=data, layout=layout)

py.iplot(data, filename='train-label-dist')
# Co-occurence Matrix

com = np.zeros([len(counts)]*2)

for i, l in enumerate(list(counts.keys())):

    for i2, l2 in enumerate(list(counts.keys())):

        c = 0

        cy = 0

        for row in labels.values:

            if l in row:

                c += 1

                if l2 in row: cy += 1

        com[i, i2] = cy / c



data=[go.Heatmap(z=com, x=list(counts.keys()), y=list(counts.keys()))]

layout=go.Layout(height=800, width=800, title='Co-occurence matrix of training labels')

fig=dict(data=data, layout=layout)

py.iplot(data, filename='train-com')
import cv2



new_style = {'grid': False}

plt.rc('axes', **new_style)

_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(20, 20))

i = 0

for f, l in df_train[:9].values:

    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))

    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    ax[i // 3, i % 3].set_title('{} - {}'.format(f, l))

    #ax[i // 4, i % 4].show()

    i += 1

    

plt.show()