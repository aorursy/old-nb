#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import time
startTime = time.time()
# Any results you write to the current directory are saved as output.




import pandas as pd
import numpy as np
import operator 
import re
import gc
import pickle

from keras.preprocessing.text import Tokenizer




np.random.seed(42)




def load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o) > 100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index




def index_embs(embeddings_index, word_index, NUM_WORDS, fileName):
    
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    embedding_matrix = np.random.normal(emb_mean, emb_std, (NUM_WORDS, embed_size))

    for word, i in word_index.items():
        if i >= NUM_WORDS: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    np.save(fileName, embedding_matrix)
    print(fileName + " embedding matrix saved!")
    
    del(embeddings_index)
    del(word_index)
    del(embedding_matrix)
    gc.collect()




def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words




def build_vocab(texts, num_words):
#     sentences = texts.apply(lambda x: x.split()).values
#     vocab = {}
#     for sentence in sentences:
#         for word in sentence:
#             try:
#                 vocab[word] += 1
#             except KeyError:
#                 vocab[word] = 1
    tokenizer = Tokenizer(num_words=NUM_WORDS, filters="")
    tokenizer.fit_on_texts(texts)
    
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("\nKeras text tokenizer has been saved as tokenizer.pickle")
    
    return tokenizer.word_index




def add_lower(embedding, vocab):
    """
    Therer are words that are known with upper letters and unknown without. This method saves both variants of the word
    
    """

    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")




def known_contractions(embed):
    
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known




def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text




def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown




def clean_special_chars(text, punct):
#     for p in mapping:
#         text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text




def get_mappings():
    """
    returns: 
    mispell_dict: mapping from mispelled word to correct word
    contraction_mapping: mapping from contraction to full word(s)
    punct_mapping: mapping from punctuation/special char to proper punctuation
    """
    mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
    punct = ":)(-!?|;\'$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√"
#     punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
#     punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

    return mispell_dict, contraction_mapping, punct #punct_mapping, punct




def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x




def run_text_preprocessing(df, textCol, outputFileName):
    mispell_map, contraction_map, punct = get_mappings()
    
#     print("\nWorking on: " + outputFileName)
#     print("\n****** WORD COVERAGE BEFORE PREPROCESSING ******")
#     vocab = build_vocab(df[textCol])
#     print("Glove : ")
#     oov_glove = check_coverage(vocab, embed_glove)
#     print("Paragram : ")
#     oov_paragram = check_coverage(vocab, embed_paragram)
#     print("FastText : ")
#     oov_fasttext = check_coverage(vocab, embed_fasttext)
      
    # To lower char
    df['lowered_question'] = df[textCol].apply(lambda x: x.lower())
    # Contractions
    df['treated_question'] = df['lowered_question'].apply(lambda x: clean_contractions(x, contraction_map))
    # Punct + Special chars
    df['treated_question'] = df['treated_question'].apply(lambda x: clean_special_chars(x, punct))
    # Mispelling
    #df['treated_question'] = df['treated_question'].apply(lambda x: correct_spelling(x, mispell_map))
    
#     print("\n****** FINAL WORD COVERAGE AFTER PREPROCESSING ******")
#     vocab = build_vocab(df["treated_question"])
#     print("Glove : ")
#     oov_glove = check_coverage(vocab, embed_glove)
#     print("Paragram : ")
#     oov_paragram = check_coverage(vocab, embed_paragram)
#     print("FastText : ")
#     oov_fasttext = check_coverage(vocab, embed_fasttext)
    
    print("\nDATAFRAME SAVED TO:")
    print("{0}_processed_text.csv".format(outputFileName))
    df.to_csv("{0}_processed_text.csv".format(outputFileName))
    
    del(df)
    del(mispell_map)
    del(contraction_map) 
    del(punct)
    gc.collect()




train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")




run_text_preprocessing(train, "question_text", "train")
run_text_preprocessing(test, "question_text", "test")




df_texts = pd.concat([train.drop('target', axis=1),test])
df_texts.to_csv("texts_processed.csv")




del(train)
del(test)
gc.collect()




glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'

NUM_WORDS = None
word_idx = build_vocab(df_texts["treated_question"], NUM_WORDS)
NUM_WORDS = len(word_idx) + 1
print("\n\n****** LOAD EMBEDDINGS ******")

print("\n---Extracting GloVe embedding---")
embed_glove = load_embed(glove)
index_embs(embed_glove, word_idx, NUM_WORDS, fileName="glove")
del(embed_glove)
gc.collect()

# print("\n---Extracting Paragram embedding---")
# embed_paragram = load_embed(paragram)
# index_embs(embed_paragram, word_idx, NUM_WORDS, fileName="paragram")
# del(embed_paragram)
# gc.collect()

print("\n---Extracting FastText embedding---")
embed_fasttext = load_embed(wiki_news)
index_embs(embed_fasttext, word_idx, NUM_WORDS, fileName="fasttext")
del(embed_fasttext)
gc.collect()




print(os.listdir())
(time.time() - startTime)/ 60




# dump all variables
import gc
gc.collect()
get_ipython().run_line_magic('reset', '-f')




import pandas as pd
import numpy as np
import gc
import time
import pickle
import six
import copy
from six.moves import zip
import warnings

from tensorflow import set_random_seed

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, f1_score

from keras.engine.topology import Layer
from keras.utils.generic_utils import serialize_keras_object
from keras.utils.generic_utils import deserialize_keras_object
from keras.legacy import interfaces
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Embedding, GRU, TimeDistributed, Dense, CuDNNGRU, Bidirectional, Dropout, SpatialDropout1D
from keras.layers import concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, BatchNormalization, CuDNNLSTM
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import Callback
from keras.optimizers import Optimizer
from keras import initializers
from keras import regularizers

import matplotlib.pyplot as plt

import time
startTimePt2 = time.time()




NUM_WORDS = None
EMBEDDING_DIM = 300
NUM_FILTERS = 50
MAX_LEN = 64 #256
BATCH_SIZE = 2560
RANDOM_STATE = 42
NUM_EPOCH = 11
LR = 0.001 # 3e-4
LR_MAX = LR * 6 # 7e-2
WD = 0.011 * (BATCH_SIZE / 979591 / NUM_EPOCH)**0.5
STEP_SIZE_CLR = 2 * (979591 / BATCH_SIZE)

# Suggested weight decay factor from the paper: w = w_norm * (b/B/T)**0.5
# b: batch size
# B: total number of training points per epoch
# T: total number of epochs
# w_norm: designed weight decay factor (w is the normalized one).




np.random.seed(RANDOM_STATE)
set_random_seed(RANDOM_STATE)




df_train = pd.read_csv("train_processed_text.csv")
df_test = pd.read_csv("test_processed_text.csv")
print("Train shape : ", df_train.shape)
print("Test shape : ", df_test.shape)




## fill up the missing values
X_train = df_train["treated_question"].fillna("_na_").values
X_test = df_test["treated_question"].fillna("_na_").values

y_train = df_train['target'].values




# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)




X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
print("hello word")




NUM_WORDS = len(tokenizer.word_index)
NUM_WORDS




import sys
def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))




## Pad the sentences 
X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_test = pad_sequences(X_test, maxlen=MAX_LEN)
print("hello word")




X_train[0].shape




del(df_train)
del(df_test)
del(tokenizer)




embedding_matrix_1 = np.load("glove.npy")
embedding_matrix_2 = np.load("fasttext.npy")
#embedding_matrix_3 = np.load("paragram.npy")
print("Done...")




embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2], axis = 0)
np.shape(embedding_matrix)




del(embedding_matrix_1)
del(embedding_matrix_2)
#del(embedding_matrix_3)




gc.collect()
print("done")




#slice sequences into many subsequences
def SliceDataSRNN(data):
    slicedData = []
    
    for i in range(data.shape[0]):
        split1 = np.split(data[i], 2)
        a=[]
        
        for j in range(2):
            s=np.split(split1[j], 2)
            a.append(s)
        slicedData.append(a)
    
    arr = np.array(slicedData, dtype=np.int32)
    
    del(slicedData)
    del(a)
    del(s)
    del(split1)    
    gc.collect()
    
    return arr




x_train_padded_seqs_split = SliceDataSRNN(X_train)
x_test_padded_seqs_split = SliceDataSRNN(X_test)




x_test_padded_seqs_split[0]




del(X_test)
del(X_train)

gc.collect()
print("done")




class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

#         self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
#         self.history.setdefault('iterations', []).append(self.trn_iterations)

#         for k, v in logs.items():
#             self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())




def f1(y_true, y_pred):
    '''
    metric from here 
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        
        return recall
    
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))




class AdamW(Optimizer):
    """Adam optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.
        weight_decay: float >= 0. Decoupled weight decay over each update.
    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/index.html)
        - [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, weight_decay=1e-4,  # decoupled weight decay (1/6)
                 epsilon=1e-8, decay=0., **kwargs):
        super(AdamW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.init_lr = lr # decoupled weight decay (2/6)
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.wd = K.variable(weight_decay, name='weight_decay') # decoupled weight decay (3/6)
        self.epsilon = epsilon
        self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]
        wd = self.wd # decoupled weight decay (4/6)

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        eta_t = lr / self.init_lr # decoupled weight decay (5/6)

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon) - eta_t * wd * p # decoupled weight decay (6/6)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'weight_decay': float(K.get_value(self.wd)),
                  'epsilon': self.epsilon}
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




# Kernel
glorotInit = initializers.glorot_uniform(seed=RANDOM_STATE)
# Recurrent
orthoInit = initializers.Orthogonal(seed=RANDOM_STATE)





from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.


        Note: The layer has been tested with Keras 1.x

        Example:
        
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...

            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)

        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = glorotInit

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]




def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result




# Kernel
glorotInit = initializers.glorot_uniform(seed=RANDOM_STATE)
# Recurrent
orthoInit = initializers.Orthogonal(seed=RANDOM_STATE)




def srnn_model():
   embedding_layer = Embedding(NUM_WORDS,
                               EMBEDDING_DIM,
                               weights=[embedding_matrix],
                               input_length=16,
                               trainable=False)
   ## ENCODER 1
   input1 = Input(shape=(16, ), dtype='int32')
   embed = embedding_layer(input1)
   drop1 = SpatialDropout1D(0.2)(embed)
   gru1 = Bidirectional(CuDNNLSTM(NUM_FILTERS, return_sequences=True, kernel_initializer=glorotInit,
                                 recurrent_initializer=glorotInit))(drop1)
   gru1_2 = Bidirectional(CuDNNLSTM(NUM_FILTERS, return_sequences=True, kernel_initializer=glorotInit,
                                 recurrent_initializer=glorotInit))(gru1)
   atten_1 = Attention()(gru1)
   atten_1_2 = Attention()(gru1_2)
   max_pool1 = GlobalMaxPooling1D()(gru1_2)
   avg_pool1 = GlobalAveragePooling1D()(gru1_2)
   conc1 = concatenate([atten_1, atten_1_2, max_pool1, avg_pool1])    
   
   Encoder1 = Model(input1, conc1)

   ## ENCODER 2
   input2 = Input(shape=(2, 16, ), dtype='int32')
   embed2 = TimeDistributed(Encoder1)(input2)
   gru2 = Bidirectional(CuDNNGRU(NUM_FILTERS, return_sequences=False, kernel_initializer=glorotInit,
                                 recurrent_initializer=glorotInit))(embed2)
#     atten_2 = Attention(2)(gru2)
#     max_pool2 = GlobalMaxPooling1D()(gru2)    
#     conc2 = concatenate([atten_2, max_pool2])    
   Encoder2 = Model(input2, gru2)

   ## ENCODER 3
   input3 = Input(shape=(2, 2, 16), dtype='int32')
   embed3 = TimeDistributed(Encoder2)(input3)
   gru3 = Bidirectional(CuDNNGRU(NUM_FILTERS, return_sequences=False, kernel_initializer=glorotInit,
                                 recurrent_initializer=glorotInit))(embed3)
#     atten_3 = Attention(2)(gru3)
#     max_pool3 = GlobalMaxPooling1D()(gru3)    
#     avg_pool3 = GlobalAveragePooling1D()(gru3)
#     conc3 = concatenate([atten_3, max_pool3, avg_pool3])    

   ## OUTPUT
   dense = Dense(32, activation="relu", kernel_initializer=glorotInit)(gru3)
   drop2 = Dropout(0.1)(dense)
   output = Dense(1, activation="sigmoid")(drop2)      
   
   model = Model(input3, output)
   #model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=['acc', f1])
   model.compile(loss='binary_crossentropy', optimizer=AdamW(lr=LR, weight_decay=0.), metrics=["acc", f1])
   return model




# def model_lstm_atten_bi():
       
#     embedding_layer = Embedding(NUM_WORDS,
#                                 EMBEDDING_DIM,
#                                 weights=[embedding_matrix],
#                                 input_length=MAX_LEN,
#                                 trainable=False)
    
#     input1 = Input(shape=(MAX_LEN,))
#     embed = embedding_layer(input1)
#     drop1 = SpatialDropout1D(0.2)
#     gru1 = Bidirectional(CuDNNGRU(NUM_FILTERS, return_sequences=True, kernel_initializer=glorotInit,
#                                recurrent_initializer=glorotInit))(embed)    
# #    batch_norm1 = BatchNormalization(momentum=0.2, center=False, scale=False)(gru1)
    
#     gru2 = Bidirectional(CuDNNGRU(NUM_FILTERS, return_sequences=True, kernel_initializer=glorotInit,
#                                recurrent_initializer=glorotInit))(gru1)    
# #    batch_norm2 = BatchNormalization(momentum=0.2, center=False, scale=False)(gru2)
    
#     atten_1 = Attention(MAX_LEN)(gru1) # skip connect
#     atten_2 = Attention(MAX_LEN)(gru2)
#     avg_pool = GlobalAveragePooling1D()(gru2)
#     max_pool = GlobalMaxPooling1D()(gru2)
    
#     conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
#     dense1 = Dense(32, activation="relu", kernel_initializer=glorotInit)(conc)
#     drop1 = Dropout(0.1)(dense1)
#     outp = Dense(1, activation="sigmoid")(drop1)    

#     model = Model(inputs=input1, outputs=outp)
#     #model.compile(loss='binary_crossentropy', optimizer="rmsprop", metrics=["acc", f1])
#     model.compile(loss='binary_crossentropy', optimizer=AdamW(lr=LR, weight_decay=WD), metrics=["acc", f1])
    
#     return model




# def model_lstm_test():
#     maxlen =  MAX_LEN
#     max_features = NUM_FILTERS
    
#     inp = Input(shape=(maxlen,))
#     x = Embedding(NUM_WORDS, 300, weights=[embedding_matrix], trainable=False)(inp)
#     x = SpatialDropout1D(0.2)(x)
#     x = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
#     y = Bidirectional(CuDNNLSTM(40, return_sequences=True))(x)
    
#     atten_1 = Attention(maxlen)(x) # skip connect
#     atten_2 = Attention(maxlen)(y)
#     avg_pool = GlobalAveragePooling1D()(y)
#     max_pool = GlobalMaxPooling1D()(y)
    
#     conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
#     conc = Dense(32, activation="relu")(conc)
#     conc = Dropout(0.1)(conc)
#     outp = Dense(1, activation="sigmoid")(conc)    

#     model = Model(inputs=inp, outputs=outp)
#     model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=["acc", f1])
    
#     return model




# clr = CyclicLR(base_lr=LR, max_lr=LR_MAX,
#                step_size=STEP_SIZE_CLR, mode='exp_range',
#                gamma=0.99994)
clr = CyclicLR(base_lr=0.001, max_lr=0.004,
               step_size=300., mode='exp_range',
               gamma=0.99994)




# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
def train_pred(model, train_X, train_y, val_X, val_y, test_data, callbacks=None):
    st = time.time()      
    model.fit(np.array(train_X), train_y, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, validation_data=(np.array(val_X), val_y), 
              callbacks = callbacks, verbose=2)
    
    pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)

    best_thresh = threshold_search(val_y, pred_val_y)
    
    print("\tVal F1 Score: {:.4f}\tThresh: {:.2f}".format(best_thresh["f1"], best_thresh["threshold"]))
    best_score = best_thresh["f1"]
    pred_test_y = model.predict(test_data, batch_size=1024, verbose=0)
    print("Training time was: " + str(time.time() - st))
    print('=' * 60)
    
    return pred_val_y, pred_test_y, best_score




# Normal Bi-LSTM

# train_meta = np.zeros(y_train.shape)
# test_meta = np.zeros(len(X_test))

# splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE).split(X_train, y_train))

# for idx, (train_idx, valid_idx) in enumerate(splits):
#     X_train_tmp = X_train[train_idx]
#     y_train_tmp = y_train[train_idx]
#     X_val_tmp =  X_train[valid_idx]
#     y_val_tmp = y_train[valid_idx]
    
#     #model = srnn_model()
#     #model = model_lstm_atten_bi()
#     model = model_lstm_test()
#     pred_val_y, pred_test_y, best_score = train_pred(model, X_train_tmp, y_train_tmp, X_val_tmp, y_val_tmp, X_test, 
#                                                      callbacks=[clr])
    
#     train_meta[valid_idx] = pred_val_y.reshape(-1)
#     test_meta += pred_test_y.reshape(-1) / len(splits)

#     del(X_train_tmp)
#     del(y_train_tmp)
#     del(X_val_tmp)
#     del(y_val_tmp)
#     gc.collect()




# Sliced RNN with Attention

train_meta = np.zeros(y_train.shape)
test_meta = np.zeros(len(x_test_padded_seqs_split))

splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE).split(x_train_padded_seqs_split, y_train))

for idx, (train_idx, valid_idx) in enumerate(splits):
    X_train_tmp = x_train_padded_seqs_split[train_idx]
    y_train_tmp = y_train[train_idx]
    X_val_tmp =  x_train_padded_seqs_split[valid_idx]
    y_val_tmp = y_train[valid_idx]
    
    model = srnn_model()
    #model = model_lstm_atten_bi()
    #model = model_lstm_test()
    pred_val_y, pred_test_y, best_score = train_pred(model, X_train_tmp, y_train_tmp, X_val_tmp, y_val_tmp, x_test_padded_seqs_split, 
                                                     callbacks=[clr])
    
    train_meta[valid_idx] = pred_val_y.reshape(-1)
    test_meta += pred_test_y.reshape(-1) / len(splits)

    del(X_train_tmp)
    del(y_train_tmp)
    del(X_val_tmp)
    del(y_val_tmp)
    gc.collect()




search_result = threshold_search(y_train, train_meta)
print(search_result)

sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = test_meta > search_result['threshold']
sub.to_csv("submission.csv", index=False)




(time.time() - startTimePt2) / 60











