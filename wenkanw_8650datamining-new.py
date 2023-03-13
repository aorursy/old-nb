#!/usr/bin/env python
# coding: utf-8



# !pip install tensorflow-cpu==2.1.0rc1
get_ipython().system('pip install tensorflow-gpu==2.1.0rc1')




import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import operator
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras import layers

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)




print(tf.__version__)




x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))

print("Device name: {}".format((x.device)))




print(tf.executing_eagerly())




print(tf.keras.__version__)




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results

# df = pd.read_csv("train.csv")
df  = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")
df_test = pd.read_csv('/kaggle/input/quora-insincere-questions-classification/test.csv')




df[df.target==1].head(10)




print("Number of questions: ", df.shape[0])




df.target.value_counts()




print("Percentage of insincere questions: {}".format(sum(df.target == 1)*100/len(df.target)))




df.isna().sum()




# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  




plot_wordcloud(df[df.target == 0]["question_text"], title="Word Cloud of Sincere Questions")




plot_wordcloud(df[df.target == 1]["question_text"], title="Word Cloud of Insincere Questions")




stopwords = set(STOPWORDS)




sincere_words = df[df.target==0].question_text.apply(lambda x: x.lower().split()).tolist()
insincere_words = df[df.target==1].question_text.apply(lambda x: x.lower().split()).tolist()

sincere_words = [item for sublist in sincere_words for item in sublist if item not in stopwords]
insincere_words = [item for sublist in insincere_words  for item in sublist if item not in stopwords ]









print('Number of sincere words',len(sincere_words))
print('Number of insincere words',len(insincere_words))




sincere_words_counter = Counter(sincere_words)
insincere_words_counter = Counter(insincere_words)
print(sincere_words_counter,insincere_words_counter)




most_common_sincere_words = sincere_words_counter.most_common()[:10]
most_common_sincere_words = pd.DataFrame(most_common_sincere_words)
most_common_sincere_words.columns = ['word', 'freq']
most_common_sincere_words['percentage'] = most_common_sincere_words.freq *100 / sum(most_common_sincere_words.freq)
most_common_sincere_words




most_common_insincere_words = insincere_words_counter.most_common()[:10]
most_common_insincere_words = pd.DataFrame(most_common_insincere_words)
most_common_insincere_words.columns = ['word', 'freq']
most_common_insincere_words['percentage'] = most_common_insincere_words.freq *100 / sum(most_common_insincere_words.freq)
most_common_insincere_words




def generate_ngrams(words, n):
    
    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[words[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]




n = 3




sincere_ngram_counter = Counter(generate_ngrams(sincere_words, n))
insincere_ngram_counter = Counter(generate_ngrams(insincere_words, n))




most_common_sincere_ngram = sincere_ngram_counter.most_common()[:10]
most_common_sincere_ngram = pd.DataFrame(most_common_sincere_ngram)
most_common_sincere_ngram.columns = ['word', 'freq']
most_common_sincere_ngram['percentage'] = most_common_sincere_ngram.freq *100 / sum(most_common_sincere_ngram.freq)
most_common_sincere_ngram




most_common_insincere_ngram = insincere_ngram_counter.most_common()[:10]
most_common_insincere_ngram = pd.DataFrame(most_common_insincere_ngram)
most_common_insincere_ngram.columns = ['word', 'freq']
most_common_insincere_ngram['percentage'] = most_common_insincere_ngram.freq *100 / sum(most_common_insincere_ngram.freq)
most_common_insincere_ngram




# config values
embed_size = 300 # how big is each word vector
max_features = 50000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 100 # max number of words in a question to use




X_train, X_test  = train_test_split(df, test_size=0.1, random_state=2019)
y_train, y_test = X_train['target'].values, X_test['target'].values




X_train.shape




X_train = X_train['question_text'].fillna('_NA_').values
X_test = X_test['question_text'].fillna('_NA_').values
X_submission = df_test['question_text'].fillna('_NA_').values




X_train.shape




tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
X_submission = tokenizer.texts_to_sequences(X_submission)




X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
X_submission = pad_sequences(X_submission, maxlen=maxlen)




X_test.shape




#Convert to TF-IDF data
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer= TfidfTransformer().fit(X_train)    
x_train = tf_transformer.transform(X_train)
tf_transformer.fit(X_test)
x_test = tf_transformer.transform(X_test)




print(x_test[0])




from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import numpy as np
corpus = ['this is the first document',
           'this document is the second document',
          'and this is the third one',
           'is this the first document']
vocabulary = ['this', 'document', 'first', 'is', 'second', 'the',
              'and', 'one']
# pipe['count'].transform(corpus).toarray()
print(len(pipe['tfid'].idf_))
# print(X_train)

# pipe['count'].transform(corpus).toarray()
print(len(pipe1['tfid'].idf_))
# print(X_test)
print(maxlen)


# x_train = pad_sequences(pipe['tfid'].idf_, maxlen=maxlen)
# x_test = pad_sequences(pipe1['tfid'].idf_, maxlen=maxlen)
# print(x_test.shape)
# print(x_train.shape)




def data_prep(df):
    print("Splitting dataframe with shape {} into training and test datasets".format(df.shape))
    X_train, X_test  = train_test_split(df, test_size=0.1, random_state=2019)
    y_train, y_test = X_train['target'].values, X_test['target'].values
    
    print("Filling missing values")
    X_train = X_train['question_text'].fillna('_NA_').values
    X_test = X_test['question_text'].fillna('_NA_').values
    X_submission = df_test['question_text'].fillna('_NA_').values
    
    print("Tokenizing {} questions into words".format(df.shape[0]))
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    X_submission = tokenizer.texts_to_sequences(X_submission)
    
    print("Padding sequences for uniform dimensions")
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)
    X_submission = pad_sequences(X_submission, maxlen=maxlen)
    
    print("Completed data preparation, returning training, test and submission datasets, split as dependent(X) and independent(Y) variables")
    
    return X_train, X_test, y_train, y_test, X_submission




model1 = Sequential()
model1.add(Embedding(max_features, embed_size, input_length=maxlen))
# model1.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
model1.add(GlobalMaxPool1D())
model1.add(Dropout(0.2))
model1.add(Dense(64, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(16, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model1.summary()




len(y_train[y_train==0]),len(y_train[y_train==1])





insincere_x = X_train[y_train==1]
sincere_x = X_train[y_train==0]
insincere_y = y_train[y_train==1]
sincere_y = y_train[y_train==0]
insincere_x.shape,sincere_x.shape




import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)  
#         print ('neighbors',neighbors)
        for i in range(len(self.samples)):
#             print('samples',self.samples[i])
            nnarray=neighbors.kneighbors(self.samples[i].reshape((1,-1)),return_distance=False)[0]  #Finds the K-neighbors of a point.
#             print ('nna',nnarray)
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class sample i ,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
#             print('j',j)
            nn=random.randint(0,self.k-1) 
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
#             print(self.synthetic)
            
a=np.array([[1,2,3],[4,5,6],[2,3,1],[2,1,2],[2,3,4],[2,3,4]])
s=Smote(a,N=200)
n= s.over_sampling()
print(np.shape(n))




def balance_data(sin_i,sin_t, in_sin_i,in_sin_t,method ='downsampling', test_sin_sample= 0,test_insin_sample= 0,sample_size=140000):
    """

    :param aep_i:
    :param aep_t:
    :param non_aep_i:
    :param non_aep_t:
    :param method:
    :param test_aep_sample:
    :param test_nonaep_sample:
    :return:
    """

    # split test set and train set    
    sin_test_i = sin_i[0:test_sin_sample,:]
    sin_test_t = sin_t[0:test_sin_sample]
    test_insin_sample = int(0.3*len(in_sin_i))
    in_sin_test_i = in_sin_i[0:test_insin_sample,:]
    in_sin_test_t = in_sin_t[0:test_sin_sample]

    # test set
    test_i = np.concatenate((sin_test_i,in_sin_test_i),axis=0)
    test_t = []
    test_t.extend(sin_test_t)
    test_t.extend( in_sin_test_t)

    # train set
    sin_i =sin_i[test_sin_sample:,:]
    sin_t = np.array(sin_t[test_sin_sample:])
    in_sin_i = in_sin_i[test_insin_sample:, :]
    in_sin_t = in_sin_t[test_insin_sample:]

 
    # balance training set here
    if method == 'downsampling':
        sample_range = len(sin_i)
        indices = np.random.randint(sample_range, size=len(in_sin_i))
        new_sin_i = sin_i[indices,:]
        new_sin_t = sin_t[indices]
        
        train_i = np.concatenate((new_sin_i, in_sin_i),axis=0)
        train_t =[]
        train_t.extend(new_sin_t)
        train_t.extend(in_sin_t)


    elif method == 'oversampling':
        sample_range = len(in_sin_i)
        indices = np.random.randint(sample_range, size=len(sin_i))
        new_insin_i = in_sin_i[indices,:]
        new_insin_t = in_sin_t[indices]
        
        train_i = np.concatenate((sin_i,new_insin_i),axis=0)
        train_t =[]
        train_t.extend(sin_t)
        train_t.extend(new_insin_t)
        pass
    elif method == 'SMOTE':
        
#         s= Smote(sin_i,N=10*int(5000/len(insin_i)))
        s= Smote(sin_i,N=200)
        new_insin_i = s.over_sampling()
        new_insin_t = [1.0]*len(new_sin_i)
        indices = np.random.randint(len(sin_i), size=len(new_insin_i))
        new_sin_i = sin_i[indices,:]
        new_sin_t = sin_t[indices]
        
        train_i = np.concatenate((new_sin_i,new_insin_i),axis=0)
        train_t =[]
        train_t.extend(new_sin_t)
        train_t.extend(new_insin_t)
    else:
#         sample_size = 5000
        indices = np.random.randint(len(in_sin_i), size=sample_size)
        new_insin_i = in_sin_i[indices,:]
        new_insin_t = in_sin_t[indices]
        indices = np.random.randint(len(sin_i), size=sample_size)
        new_sin_i = sin_i[indices,:]
        new_sin_t = sin_t[indices]
        
        train_i = np.concatenate((new_sin_i,new_insin_i),axis=0)
        train_t =[]
        train_t.extend(new_sin_t)
        train_t.extend(new_insin_t)
        pass


    return train_i,train_t

new_x,new_y = balance_data(sincere_x,sincere_y, insincere_x,insincere_y,method ='', test_sin_sample= 0,test_insin_sample= 0,sample_size=140000)
len(new_x), len(new_y)




X_test.shape




new_sincere_y = np.array(sincere_y)
new_sincere_y[new_sincere_y==0].shape, new_sincere_y[new_sincere_y==1].shape




X_train = np.array(new_x)
y_train = np.array(new_y)




X_train.shape




def get_fscore_matrix(fitted_clf, model_name):
    print(model_name, ' :')
    
    # get classes predictions for the classification report 
    y_train_pred, y_pred = fitted_clf.predict(X_train), fitted_clf.predict(X_test)
    print(classification_report(y_test, y_pred), '\n') # target_names=y
    
    # computes probabilities keep the ones for the positive outcome only      
    print(f'F1-score = {f1_score(y_test, y_pred):.2f}')




from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
lrmodel = LogisticRegression(class_weight={0:y_train.sum(), 1:len(y_train) - y_train.sum()}, C=0.5, max_iter=100, n_jobs=-1)
lrmodel.fit(X_train, y_train)
get_fscore_matrix(lrmodel, 'LogisticRegression')




from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(X_train, y_train)
get_fscore_matrix(dtc, 'DecisionTreeClassifier')




from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
get_fscore_matrix(gnb, 'Gaussian Naive Bayes Classifier')




get_ipython().run_line_magic('time', 'model1.fit(X_train, y_train, batch_size=512, epochs=2, validation_data=(X_test, y_test), verbose = 1)')




pred_test_y = model1.predict([X_test], batch_size=1024, verbose=1)




opt_prob = None
f1_max = 0

for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(y_test, (pred_test_y > thresh).astype(int))
    print('F1 score at threshold {} is {}'.format(thresh, f1))
    
    if f1 > f1_max:
        f1_max = f1
        opt_prob = thresh
        
#print('Optimal probabilty threshold is {} for maximum F1 score {}'.format(opt_prob, f1_max))
print('The F1 score is {}'.format(f1_max))




model2 = Sequential()
model2.add(Embedding(max_features, embed_size, input_length=maxlen))
model2.add(Bidirectional(LSTM(128, return_sequences=True)))
model2.add(GlobalMaxPool1D())
model2.add(Dropout(0.2))
# model2.add(Dense(64, activation='relu'))
# model2.add(Dropout(0.2))
model2.add(Dense(32, activation='relu'))
model2.add(Dropout(0.2))
# model2.add(Dense(16, activation='relu'))
# model2.add(Dropout(0.2))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model2.summary()




get_ipython().run_line_magic('time', 'model2.fit(X_train, y_train, batch_size=1024, epochs=1, validation_data=(X_test, y_test), verbose = 1)')




pred_submission_y = model1.predict([X_submission], batch_size=1024, verbose=1)
pred_submission_y = (pred_submission_y > opt_prob).astype(int)

df_submission = pd.DataFrame({'qid': df_test['qid'].values})
df_submission['prediction'] = pred_submission_y
#df_submission.to_csv("submission.csv", index=False)




def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding="utf8") if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index




glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'




print("Extracting GloVe embedding")
embed_glove = load_embed(glove)
#print("Extracting Paragram embedding")
#embed_paragram = load_embed(paragram)
#print("Extracting FastText embedding")
#embed_fasttext = load_embed(wiki_news)




def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab




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




vocab = build_vocab(df['question_text'])




print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
#print("Paragram : ")
#oov_paragram = check_coverage(vocab, embed_paragram)
#print("FastText : ")
#oov_fasttext = check_coverage(vocab, embed_fasttext)




type(embed_glove)




dict(list(embed_glove.items())[20:22])




df['processed_question'] = df['question_text'].apply(lambda x: x.lower())




vocab_low = build_vocab(df['processed_question'])




print("Glove : ")
oov_glove = check_coverage(vocab_low, embed_glove)
#print("Paragram : ")
#oov_paragram = check_coverage(vocab_low, embed_paragram)
#print("FastText : ")
#oov_fasttext = check_coverage(vocab_low, embed_fasttext)




oov_glove[1:20]




def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")




print("Glove : ")
add_lower(embed_glove, vocab)
#print("Paragram : ")
#add_lower(embed_paragram, vocab)
#print("FastText : ")
#add_lower(embed_fasttext, vocab)




print("Glove : ")
oov_glove = check_coverage(vocab_low, embed_glove)
#print("Paragram : ")
#oov_paragram = check_coverage(vocab_low, embed_paragram)
#print("FastText : ")
#oov_fasttext = check_coverage(vocab_low, embed_fasttext)




oov_glove[1:20]




punctuations = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'




def clean_text(x):

    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in punctuations:
        x = x.replace(punct, '')
    return x




df["processed_question"] = df["processed_question"].progress_apply(lambda x: clean_text(x))




vocab_low = build_vocab(df['processed_question'])




print("Glove : ")
oov_glove = check_coverage(vocab_low, embed_glove)
#print("Paragram : ")
#oov_paragram = check_coverage(vocab_low, embed_paragram)
#print("FastText : ")
#oov_fasttext = check_coverage(vocab_low, embed_fasttext)




df['question_text'] = df['processed_question']




X_train, X_test, y_train, y_test, X_submission = data_prep(df)




model1 = Sequential()
model1.add(Embedding(max_features, embed_size, input_length=maxlen, weights = [embed_glove]))
model1.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
model1.add(GlobalMaxPool1D())
model1.add(Dropout(0.2))
model1.add(Dense(64, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(32, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.summary()
X_train, X_test, y_train, y_test = train_test_split(df_train.text, df_train.target, test_size=0.1, random_state=1)

prediction = pipeline.predict(X_test)
print(metrics.accuracy_score(y_test, prediction))
print(metrics.precision_score(y_test, prediction))




get_ipython().run_line_magic('time', 'model1.fit(X_train, y_train, batch_size=512, epochs=5, validation_data=(X_test, y_test), verbose = 1)')




pred_test_y = model1.predict([X_test], batch_size=1024, verbose=1)




opt_prob = None
f1_max = 0

for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    f1 = metrics.f1_score(y_test, (pred_test_y > thresh).astype(int))
    print('F1 score at threshold {} is {}'.format(thresh, f1))
    
    if f1 > f1_max:
        f1_max = f1
        opt_prob = thresh
        
print('Optimal probabilty threshold is {} for maximum F1 score {}'.format(opt_prob, f1_max))




pred_submission_y = model1.predict([X_submission], batch_size=1024, verbose=1)
pred_submission_y = (pred_submission_y > opt_prob).astype(int)

df_submission = pd.DataFrame({'qid': df_test['qid'].values})
df_submission['prediction'] = pred_submission_y
df_submission.to_csv("submission.csv", index=False)






