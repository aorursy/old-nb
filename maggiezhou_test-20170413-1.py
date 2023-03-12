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
import pandas as pd

import numpy as np

import nltk

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer

import re

from string import punctuation
train = pd.read_csv("../input/train.csv")[:100]

test = pd.read_csv("../input/test.csv")[:100]

# Check for any null values

print(train.isnull().sum())

print(test.isnull().sum())
