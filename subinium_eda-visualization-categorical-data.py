#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_palette('bright')
import os
print(os.listdir('../input/cat-in-the-dat'))

train_df = pd.read_csv('../input/cat-in-the-dat/train.csv')
test_df = pd.read_csv('../input/cat-in-the-dat/test.csv')
print(train_df.shape, test_df.shape)

train_df.head(3)

train_df.describe(include='O')

sns.heatmap(train_df.corr())

sns.countplot(train_df['target'])
print("percentage of target 1 : {}%".format(train_df['target'].sum() / len(train_df)))

print(train_df['month'].value_counts().sort_index(axis = 0) )
print(train_df['day'].value_counts().sort_index(axis = 0) )

def percentage_of_feature_target(df, feat, tar, tar_val):
    return df[df[tar]==tar_val][feat].value_counts().sort_index(axis = 0) / df[feat].value_counts().sort_index(axis = 0)

P_month = percentage_of_feature_target(train_df, 'month', 'target', 1)

P_month.plot()

fig, ax = plt.subplots(1,1, figsize=(20, 5))
sns.countplot('month', hue='target', data= train_df, ax=ax)
plt.show()

print(percentage_of_feature_target(train_df, 'day', 'target', 1))

fig, ax = plt.subplots(1,1, figsize=(20, 5))
sns.countplot('day', hue='target', data= train_df, ax=ax)
plt.show()

fig, ax = plt.subplots(1,4, figsize=(20, 5))
for i in range(4):
    sns.countplot(f'bin_{i}', hue='target', data= train_df, ax=ax[i])
    ax[i].set_title(f'bin_{i} feature countplot')
    print(percentage_of_feature_target(train_df,f'bin_{i}','target',1))
plt.show()

for i in range(9):
    tmp_list = train_df[f'nom_{i}'].value_counts().index
    print(f'nom_{i} feature\'s unique value {len(tmp_list)}')

for i in range(5):
    print(percentage_of_feature_target(train_df, f'nom_{i}', 'target',1))

fig, ax = plt.subplots(1,5, figsize=(25, 7))
for i in range(5):
    sns.countplot(f'nom_{i}', hue='target', data= train_df, ax=ax[i])
    plt.setp(ax[i].get_xticklabels(),rotation=30)
plt.show()

print(train_df['nom_5'].value_counts()) 

fig, ax = plt.subplots(4,1,figsize=(40, 40))
for i in range(5, 9):
    sns.countplot(sorted(train_df[f'nom_{i}']), ax=ax[i-5])
    plt.setp(ax[i-5].get_xticklabels(),rotation=90)
plt.show()

for i in range(5, 9):
    fig, ax = plt.subplots(1,1,figsize=(8, 2))
    P_nom = percentage_of_feature_target(train_df, f'nom_{i}', 'target', 1)
    P_nom.plot() # easy plot
    #sns.barplot(P_nom.index, P_nom, ax=ax[i-5])
    #plt.setp(ax[i-5].get_xticklabels(),rotation=90)
plt.show()

# fig, ax = plt.subplots(1,1,figsize=(40, 10))
# sns.countplot(sorted(train_df['nom_9']), ax=ax)
# plt.show();

train_df['ord_5'].value_counts().sort_index(axis=0)

P_ord5 = percentage_of_feature_target(train_df, 'ord_5', 'target', 1)
fig, ax = plt.subplots(1,1,figsize=(20,7))
sns.barplot(P_ord5.index, P_ord5, ax=ax)
plt.title('ord_5 : Percentage of target==1 in dictionary order')
plt.setp(ax.get_xticklabels(),rotation=90, fontsize=5)
plt.show()

P_ord4 = percentage_of_feature_target(train_df, 'ord_4', 'target', 1)
fig, ax = plt.subplots(1,1,figsize=(20,7))
sns.barplot(P_ord4.index, P_ord4, ax=ax)
plt.title('ord_4 : Percentage of target==1 in dictionary order')
plt.setp(ax.get_xticklabels(),rotation=90, fontsize=5)
plt.show()

P_ord3 = percentage_of_feature_target(train_df, 'ord_3', 'target', 1)
fig, ax = plt.subplots(1,1,figsize=(20,7))
sns.barplot(P_ord3.index, P_ord3, ax=ax)
plt.title('ord_3 : Percentage of target==1 in dictionary order')
plt.setp(ax.get_xticklabels(),rotation=90, fontsize=5)
plt.show()

# import sklearn.

