#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')




dict = {} # display_id, count
df = pd.read_csv('../input/clicks_train.csv', iterator=True, chunksize=10000)
    




adcount = {}







