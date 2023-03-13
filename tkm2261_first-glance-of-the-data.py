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




get_ipython().run_cell_magic('bash', '', 'wc -l ../input/*')




pd.read_csv('../input/submissionFile', nrows=10) # There is no info what each class means. 




df_test_text = pd.read_csv('../input/test_text', nrows=10, skiprows=1, sep='\|\|', header=None, 
                           names=['ID', 'Text'], engine='python')
df_test_text




df_test_text.loc[0, 'Text']




pd.read_csv('../input/test_variants', nrows=10)




df_training_text = pd.read_csv('../input/training_text', nrows=10, skiprows=1, sep='\|\|', header=None, 
                                  names=['ID', 'Text'], engine='python')
df_training_text




df_training_text.loc[0, 'Text']




pd.read_csv('../input/training_variants', nrows=10)

