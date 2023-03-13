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




from sklearn.model_selection import train_test_split

def read_data():
    df = pd.read_csv("../input/train.csv")
    print ("Shape of base training File = ", df.shape)
    # Remove missing values and duplicates from training data
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print("Shape of base training data after cleaning = ", df.shape)
    return df

df = read_data()
df_train, df_test = train_test_split(df, test_size = 0.02)
print (df_train.head(2))
print (df_test.shape)




from collections import Counter
import matplotlib.pyplot as plt
import operator

def eda(df):
    print ("Duplicate Count = %s , Non Duplicate Count = %s" 
           %(df.is_duplicate.value_counts()[1],df.is_duplicate.value_counts()[0]))
    
    question_ids_combined = df.qid1.tolist() + df.qid2.tolist()
    
    print ("Unique Questions = %s" %(len(np.unique(question_ids_combined))))
    
    question_ids_counter = Counter(question_ids_combined)
    sorted_question_ids_counter = sorted(question_ids_counter.items(), key=operator.itemgetter(1))
    question_appearing_more_than_once = [i for i in question_ids_counter.values() if i > 1]
    print ("Count of Quesitons appearing more than once = %s" %(len(question_appearing_more_than_once)))
    
    
eda(df_train)




import re
import gensim
from gensim import corpora
from nltk.corpus import stopwords

words = re.compile(r"\w+",re.I)
stopword = stopwords.words('english')

def tokenize_questions(df):
    question_1_tokenized = []
    question_2_tokenized = []

    for q in df.question1.tolist():
        question_1_tokenized.append([i.lower() for i in words.findall(q) if i not in stopword])

    for q in df.question2.tolist():
        question_2_tokenized.append([i.lower() for i in words.findall(q) if i not in stopword])

    df["Question_1_tok"] = question_1_tokenized
    df["Question_2_tok"] = question_2_tokenized
    
    return df

def train_dictionary(df):
    
    questions_tokenized = df.Question_1_tok.tolist() + df.Question_2_tok.tolist()
    
    dictionary = corpora.Dictionary(questions_tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=10000000)
    dictionary.compactify()
    
    return dictionary
    
df_train = tokenize_questions(df_train)
dictionary = train_dictionary(df_train)
print ("No of words in the dictionary = %s" %len(dictionary.token2id))

df_test = tokenize_questions(df_test)




def get_vectors(df, dictionary):
    
    question1_vec = [dictionary.doc2bow(text) for text in df.Question_1_tok.tolist()]
    question2_vec = [dictionary.doc2bow(text) for text in df.Question_2_tok.tolist()]
    
    question1_csc = gensim.matutils.corpus2csc(question1_vec, num_terms=len(dictionary.token2id))
    question2_csc = gensim.matutils.corpus2csc(question2_vec, num_terms=len(dictionary.token2id))
    
    return question1_csc.transpose(),question2_csc.transpose()


q1_csc, q2_csc = get_vectors(df_train, dictionary)

print (q1_csc.shape)
print (q2_csc.shape)




from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics.pairwise import manhattan_distances as md
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import MinMaxScaler

minkowski_dis = DistanceMetric.get_metric('minkowski')
mms_scale_man = MinMaxScaler()
mms_scale_euc = MinMaxScaler()
mms_scale_mink = MinMaxScaler()

def get_similarity_values(q1_csc, q2_csc):
    cosine_sim = []
    manhattan_dis = []
    eucledian_dis = []
    jaccard_dis = []
    minkowsk_dis = []
    
    for i,j in zip(q1_csc, q2_csc):
        sim = cs(i,j)
        cosine_sim.append(sim[0][0])
        sim = md(i,j)
        manhattan_dis.append(sim[0][0])
        sim = ed(i,j)
        eucledian_dis.append(sim[0][0])
        i_ = i.toarray()
        j_ = j.toarray()
        try:
            sim = jsc(i_,j_)
            jaccard_dis.append(sim)
        except:
            jaccard_dis.append(0)
            
        sim = minkowski_dis.pairwise(i_,j_)
        minkowsk_dis.append(sim[0][0])
    
    return cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis    


# cosine_sim = get_cosine_similarity(q1_csc, q2_csc)
cosine_sim, manhattan_dis, eucledian_dis, jaccard_dis, minkowsk_dis = get_similarity_values(q1_csc[0:1000,:], q2_csc[0:1000,:])
print ("cosine_sim sample= \n", cosine_sim[0:2])
print ("manhattan_dis sample = \n", manhattan_dis[0:2])
print ("eucledian_dis sample = \n", eucledian_dis[0:2])
print ("jaccard_dis sample = \n", jaccard_dis[0:2])
print ("minkowsk_dis sample = \n", minkowsk_dis[0:2])

eucledian_dis_array = np.array(eucledian_dis).reshape(-1,1)
manhattan_dis_array = np.array(manhattan_dis).reshape(-1,1)
minkowsk_dis_array = np.array(minkowsk_dis).reshape(-1,1)
    
mms_scale_man.fit(manhattan_dis_array)
mms_scale_euc.fit(eucledian_dis_array)
mms_scale_mink.fit(minkowsk_dis_array)




from sklearn.metrics import log_loss

def calculate_logloss(y_true, y_pred):
    loss_cal = log_loss(y_true, y_pred)
    return loss_cal

q1_csc_test, q2_csc_test = get_vectors(df_test, dictionary)
y_pred_cos, y_pred_man, y_pred_euc, y_pred_jac, y_pred_mink = get_similarity_values(q1_csc_test, q2_csc_test)
y_true = df_test.is_duplicate.tolist()

y_pred_man_array = mms_scale_man.transform(np.array(y_pred_man).reshape(-1,1))
y_pred_man = y_pred_man_array.tolist()

y_pred_euc_array = mms_scale_euc.transform(np.array(y_pred_euc).reshape(-1,1))
y_pred_euc = y_pred_euc_array.tolist()

y_pred_mink_array = mms_scale_mink.transform(np.array(y_pred_mink).reshape(-1,1))
y_pred_mink = y_pred_mink_array.tolist()

logloss = calculate_logloss(y_true, y_pred_cos)
print ("The calculated log loss value on the test set for cosine sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_man)
print ("The calculated log loss value on the test set for manhattan sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_euc)
print ("The calculated log loss value on the test set for euclidean sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_jac)
print ("The calculated log loss value on the test set for jaccard sim is = %f" %logloss)

logloss = calculate_logloss(y_true, y_pred_mink)
print ("The calculated log loss value on the test set for minkowski sim is = %f" %logloss)

