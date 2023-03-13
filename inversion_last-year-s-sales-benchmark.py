#!/usr/bin/env python
# coding: utf-8



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




dtypes = {'store_nbr': np.dtype('int64'),
          'item_nbr': np.dtype('int64'),
          'unit_sales': np.dtype('float64'),
          'onpromotion': np.dtype('O')}

train = pd.read_csv('../input/train.csv', index_col='id', parse_dates=['date'], dtype=dtypes)
test = pd.read_csv('../input/test.csv', index_col='id', parse_dates=['date'], dtype=dtypes)
submission = pd.read_csv('../input/sample_submission.csv', index_col='id')




train.head()




test.head()




submission.head()




# We only need the corresponding dates found in Test
date_mask = (train['date'] >= '2016-08-16') & (train['date'] <= '2016-08-31')

last_year_sales = train.loc[date_mask].copy()
last_year_sales.drop('onpromotion', axis=1, inplace=True)

# Make a look-up dictionary with keys: date, store_nbr, item_nbr
last_year_sales = last_year_sales.set_index(['date', 'store_nbr', 'item_nbr'])
last_year_sales = last_year_sales.to_dict()['unit_sales']

benchmark = submission.copy()

# Use the look-up dictionary, using the .get method so we can default in a value of 0
benchmark['unit_sales'] =     test.apply(lambda x: last_year_sales.get((x['date'] - pd.Timedelta(365, unit='d'), 
                                              x['store_nbr'],
                                              x['item_nbr']), 0), axis=1)

# Unless you enjoy seeing red errors after your submission uploads
benchmark[benchmark['unit_sales'] < 0] = 0
    
# Repeat after me . . . "I will always compress my submission file for this contest"
benchmark.to_csv('last_year_sales.csv.gz', float_format='%.4g', compression='gzip')

