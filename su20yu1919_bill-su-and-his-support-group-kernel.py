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

clicks_train = pd.read_csv("../input/clicks_train.csv")
#clicks_test = pd.read_csv("../input/clicks_test.csv")
# documents_categories = pd.read_csv("../input/documents_categories.csv")
# documents_entities  = pd.read_csv("../input/documents_entities.csv")
# documents_meta = pd.read_csv("../input/documents_meta.csv")
# documents_topics = pd.read_csv("../input/documents_topics.csv")
# events = pd.read_csv("../input/events.csv")
# page_views_sample = pd.read_csv("../input/page_views_sample.csv")
# promoted_content = pd.read_csv("../input/promoted_content.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

# Any results you write to the current directory are saved as output.
clicks_train.head()

# Get Training and Testing Sets
ids = clicks_train.display_id.unique()
ids = np.random.choice(ids, size=len(ids)//10, replace=False)
valid = clicks_train[clicks_train.display_id.isin(ids)]
train = clicks_train[~clicks_train.display_id.isin(ids)]
print(valid.shape, train.shape)

# Initialize Things
reg = 10
count = train[train.clicked==1].ad_id.value_counts()
count_all = train.ad_id.value_counts()
del train

# functions
def get_prob(k):
    if k not in count:
        return 0
    # Notice the regularization parameter
    return float(count[k])/(float(count_all[k]) + reg)

def srt(x):
    ad_ids = map(int, x.split())
    ad_ids = sorted(ad_ids, key=get_prob, reverse =True)
    return " ".join(map(str, ad_ids))


## If you want to submit
sample_submission['ad_id'] = sample_submission.ad_id.apply(lambda x: srt(x))
sample_submission.to_csv("subm_reg_1.csv", index = False)






