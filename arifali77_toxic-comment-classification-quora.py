import numpy as np

import pandas as pd

from matplotlib import pyplot as plt


import seaborn as sns

import re
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.sample(5)
cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']
train_df.isnull().any()
# check missing values in numeric columns

train_df.describe()
unlabelled_in_all = train_df[(train_df['toxic']!=1) & (train_df['severe_toxic']!=1) & (train_df['obscene']!=1) & 

                            (train_df['threat']!=1) & (train_df['insult']!=1) & (train_df['identity_hate']!=1)]

print('Percentage of unlabelled comments is ', len(unlabelled_in_all)/len(train_df)*100)
# check for any 'null' comment

no_comment = train_df[train_df['comment_text'].isnull()]

len(no_comment)
test_df.head()
no_comment = test_df[test_df['comment_text'].isnull()]

len(no_comment)
# let's see the total rows in train, test data and the numbers for the various categories

print('Total rows in test is {}'.format(len(test_df)))

print('Total rows in train is {}'.format(len(train_df)))

print(train_df[cols_target].sum())
# Let's look at the character length for the rows in the training data and record these

train_df['char_length'] = train_df['comment_text'].apply(lambda x: len(str(x)))
train_df['char_length'].head()
# look at the histogram plot for text length

sns.set()

train_df['char_length'].hist()

plt.show()
data = train_df[cols_target]

data.head()
colormap = plt.cm.plasma

plt.figure(figsize=(7,7))

plt.title('Correlation of features & targets',y=1.05,size=14)

sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0,square=True,cmap='RdYlGn',

           linecolor='white',annot=True)
test_df['char_length'] = test_df['comment_text'].apply(lambda x: len(str(x)))
plt.figure()

plt.hist(test_df['char_length'])

plt.show()
def clean_text(text):

    text = text.lower()

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "cannot ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    text = re.sub('\W', ' ', text)

    text = re.sub('\s+', ' ', text)

    text = text.strip(' ')

    return text
# clean the comment_text in train_df 

train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))

train_df['comment_text'].head()
# clean the comment_text in test_df 

test_df['comment_text'] = test_df['comment_text'].map(lambda com : clean_text(com))
train_df.head()
train_df = train_df.drop('char_length',axis=1)
X = train_df.comment_text

test_X = test_df.comment_text
print(X.shape, test_X.shape)
# import and instantiate TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(max_features=5000,stop_words='english')

vect
# learn the vocabulary in the training data, then use it to create a document-term matrix

X_dtm = vect.fit_transform(X)

# examine the document-term matrix created from X_train

X_dtm
# transform the test data using the earlier fitted vocabulary, into a document-term matrix

test_X_dtm = vect.transform(test_X)

# examine the document-term matrix from X_test

test_X_dtm
# import and instantiate the Logistic Regression model

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

logreg = LogisticRegression(C=12.0)



# create submission file

submission_binary = pd.read_csv('../input/sample_submission.csv')



for label in cols_target:

    print('... Processing {}'.format(label))

    y = train_df[label]

    # train the model using X_dtm & y

    logreg.fit(X_dtm, y)

    # compute the training accuracy

    y_pred_X = logreg.predict(X_dtm)

    print('Training accuracy is {}'.format(accuracy_score(y, y_pred_X)))

    # compute the predicted probabilities for X_test_dtm

    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]

    submission_binary[label] = test_y_prob
submission_binary.head()
# generate submission file

submission_binary.to_csv('submission_binary.csv',index=False)
# create submission file

submission_chains = pd.read_csv('../input/sample_submission.csv')



# create a function to add features

def add_feature(X, feature_to_add):

    '''

    Returns sparse feature matrix with added feature.

    feature_to_add can also be a list of features.

    '''

    from scipy.sparse import csr_matrix, hstack

    return hstack([X, csr_matrix(feature_to_add).T], 'csr')
for label in cols_target:

    print('... Processing {}'.format(label))

    y = train_df[label]

    # train the model using X_dtm & y

    logreg.fit(X_dtm,y)

    # compute the training accuracy

    y_pred_X = logreg.predict(X_dtm)

    print('Training Accuracy is {}'.format(accuracy_score(y,y_pred_X)))

    # make predictions from test_X

    test_y = logreg.predict(test_X_dtm)

    test_y_prob = logreg.predict_proba(test_X_dtm)[:,1]

    submission_chains[label] = test_y_prob

    # chain current label to X_dtm

    X_dtm = add_feature(X_dtm, y)

    print('Shape of X_dtm is now {}'.format(X_dtm.shape))

    # chain current label predictions to test_X_dtm

    test_X_dtm = add_feature(test_X_dtm, test_y)

    print('Shape of test_X_dtm is now {}'.format(test_X_dtm.shape))
submission_chains.head()
# generate submission file

submission_chains.to_csv('submission_chains.csv', index=False)
# create submission file

submission_combined = pd.read_csv('../input/sample_submission.csv')
# corr_targets = ['obscene','insult','toxic']

for label in cols_target:

    submission_combined[label] = 0.5*(submission_chains[label]+submission_binary[label])
submission_combined.head()
# generate submission file

submission_combined.to_csv('submission_combined.csv', index=False)