# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test=pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")

test.head()
# another way of parsin json file



test["event_data"] = test["event_data"].apply(lambda x : dict(json.loads(x)))

test_event_data = test["event_data"].apply(pd.Series)

test = pd.concat([test, test_event_data], axis=1)
test.shape
# remove duplicates

test = test.loc[:,~test.columns.duplicated()]
test.shape
test.head()