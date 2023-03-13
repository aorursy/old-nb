#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.




from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer




train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')




#Lets see how the data looks like
train_data




#The length of the comments vary a lot. Lets check that out
train_data['comment_text'][0]




train_data['comment_text'][5]




#Lets see max, min, std and mean of the lenths
lens = train_data.comment_text.str.len()
lens.min(), lens.max(), lens.mean(), lens.std()




lens.hist();




label_cols = ['toxic','severe_toxic', 'obscene','threat', 'insult', 'identity_hate']
train_data[label_cols].max(axis=1)
train_data['none']=1-train_data[label_cols].max(axis=1) #If every column is 0 then this statement sets none to 1
train_data #Checking the training data if the none column is updated or not. 




#Lets get rid of the empty comments. 
COMMENT='comment_text'
train_data[COMMENT].fillna("unknown",inplace =  True)
test_data[COMMENT].fillna("unknown", inplace = True)




#Lets build the model for trianing and testing. 
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r'\1', s).split()




n = train_data.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer = tokenize, 
                     min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                     smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train_data[COMMENT])
test_term_doc = vec.transform(test_data[COMMENT])
trn_term_doc, test_term_doc




def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)




from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["This is very strange", "This is very nice"]
corpus




vectorizer = TfidfVectorizer(min_df = 1)
X = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
idf
print (dict(zip(vectorizer.get_feature_names(), idf)))




x = trn_term_doc
test_x = test_term_doc




def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y)) / pr(0,y)
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y),r
    




preds = np.zeros((len(test_data), len(label_cols)))
for i,j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train_data[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]




submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)




submission




print(os.listdir("../input"))






