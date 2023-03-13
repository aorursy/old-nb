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

#import the required modules
from sklearn import preprocessing

#Load the files
gender_age_train = pd.read_csv("../input/gender_age_train.csv",encoding='utf-8')
gender_age_test = pd.read_csv("../input/gender_age_test.csv",encoding='utf-8')
device_data = pd.read_csv("../input/phone_brand_device_model.csv",encoding='utf-8')

le = preprocessing.LabelEncoder()
le.fit(device_data['phone_brand'])
le.transform(device_data['phone_brand'])

device_data['phone_brand'].value_counts()


gender_age_train


