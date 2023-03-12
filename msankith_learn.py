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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt


pylab.rcParams['figure.figsize'] = (10, 6)

limit_rows   = 3000000

df           = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,

                                                    "ind_nuevo":str,

                                                    "ult_fec_cli_1t":str,

                                                    "indext":str}, nrows=limit_rows)
df.describe()