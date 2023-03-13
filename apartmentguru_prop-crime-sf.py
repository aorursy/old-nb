import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# data visualization imports
import seaborn as sns
sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt
sf_features_train = pd.read_csv("../input/train.csv")
sf_features_train.head()


# sf_features_train.head()describe


sf_features_train.describe



