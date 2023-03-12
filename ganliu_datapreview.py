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
train = pd.read_csv("../input/train.csv", nrows=100000 )
client = pd.read_csv("../input/cliente_tabla.csv", nrows=100000)
product = pd.read_csv("../input/producto_tabla.csv", nrows=100000)
submission = pd.read_csv("../input/sample_submission.csv", nrows=100000)
town = pd.read_csv("../input/town_state.csv", nrows=100000)
train.head()
client.head()
product.head()
town.head()
submission.head()
train.describe()
