f = plt.figure()
plt.hist(train_bookings['date_time'].values, bins=100, alpha=0.5, normed=True, label='train bookings')
plt.hist(test_bookings['date_time'].values, bins=50, alpha=0.5, normed=True, label='test bookings')
plt.hist(train_clicks['date_time'].values, bins=100, alpha=0.5, normed=True, label='train clicks')
plt.title('Search time distribution')
plt.legend(loc='best')
f.savefig('SearchTime.png', dpi=300)
plt.show()


f = plt.figure()
plt.hist(train_bookings['srch_ci'].values, bins=100, alpha=0.5, normed=True, label='train bookings')
plt.hist(test_bookings['srch_ci'].dropna().values, bins=50, alpha=0.5, normed=True, label='test bookings')
plt.hist(train_clicks['srch_ci'].dropna().values, bins=100, alpha=0.5, normed=True, label='train clicks')
plt.title('Checkin time')
plt.legend(loc='best')
f.savefig('CheckinTime.png', dpi=300)
plt.show()


# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# system("ls ../input")

# Any results you write to the current directory are saved as output.

# This Julia environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/julia docker image: https://github.com/kaggle/docker-julia
# For example, here's a helpful package to load in 

# using DataFrames # data processing, CSV file I/O - e.g. readtable("../input/MyTable.csv")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

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
