import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter

def cnt(df,col):
    hist = Counter(df[col])
    for i,e in enumerate(hist):
        print df[col][i],(e)

df_train = pd.read_csv('../input/gender_age_train.csv')

cnt(df_train,"age")
cnt(df_train,"gender")
cnt(df_train,"group")

