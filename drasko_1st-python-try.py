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
# Get first 10000 rows and print some info about columns
train = pd.read_csv("../input/train.csv", parse_dates=['date_time', 'srch_ci', 'srch_co'], nrows=10000)
train.info()
import seaborn as sns

sns.countplot(y='site_name', hue='posa_continent', data=train)
# preferred continent destinations
sns.countplot(y='hotel_continent', data=train)
sns.countplot(x='hotel_cluster', hue='hotel_continent', data=train)