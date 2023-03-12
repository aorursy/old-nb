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
products=pd.read_csv('../input/train_ver2.csv',usecols=list(range(24,48,1))+[1])
products.dtypes
from math import isnan

def to_int(x):

    if isnan(x):

        return None

    else:

        return x+1
products.ind_nom_pens_ult1.astype(int)
products.ind_nom_pens_ult1=products.ind_nom_pens_ult1.apply(to_int)