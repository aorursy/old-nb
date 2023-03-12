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

train = pd.read_csv('../input/train.csv',parse_dates=['srch_ci', 'srch_co'], nrows=10000)


train.info()

import seaborn as sns
import matplotlib.pyplot as plt
# preferred continent destinations
sns.distplot(train.hotel_cluster, kde=False, color="r")


sns.FacetGrid(train, hue="hotel_cluster", size=6) \
   .map(plt.scatter, "hotel_country","hotel_continent") \
   .add_legend()

sns.pairplot(train, hue="hotel_cluster")