import numpy as np

import pandas as pd

import xgboost as xgb

import datetime

import operator

from sklearn.cross_validation import train_test_split

from collections import Counter

from nltk.corpus import stopwords

import matplotlib.pyplot as plt

from pylab import plot, show, subplot, specgram, imshow, savefig

from collections import defaultdict



RS = 12357

ROUNDS = 500

#ROUNDS = 75



print("Started")

np.random.seed(RS)

input_folder = '../input/quora-question-pairs/'



def train_xgb(X, y, params):

	print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))

	x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=RS)



	xg_train = xgb.DMatrix(x, label=y_train)

	xg_val = xgb.DMatrix(X_val, label=y_val)



	watchlist  = [(xg_train,'train'), (xg_val,'eval')]

	return xgb.train(params, xg_train, ROUNDS, watchlist)



def predict_xgb(clr, X_test):

	return clr.predict(xgb.DMatrix(X_test))



def create_feature_map(features):

	outfile = open('xgb.fmap', 'w')

	i = 0

	for feat in features:

		outfile.write('{0}\t{1}\tq\n'.format(i, feat))

		i = i + 1

	outfile.close()



def add_word_count(x, df, word):

	x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower())*1)

	x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower())*1)

	x[word + '_both'] = x['q1_' + word] * x['q2_' + word]



def main():

	params = {}

	params['objective'] = 'binary:logistic'

	params['eval_metric'] = 'logloss'

	params['eta'] = 0.11

	params['max_depth'] = 5

	params['silent'] = 1

	params['seed'] = RS



	df_train = pd.read_csv(input_folder + 'train.csv')

	df_test  = pd.read_csv(input_folder + 'test.csv')



	print("Original data: X_train: {}, X_test: {}".format(df_train.shape, df_test.shape))



	print("Features processing, be patient...")



	# If a word appears only once, we ignore it completely (likely a typo)

	# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

	def get_weight(count, eps=10000, min_count=2):

		return 0 if count < min_count else 1 / (count + eps)



	train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

	words = (" ".join(train_qs)).lower().split()

	counts = Counter(words)

	weights = {word: get_weight(count) for word, count in counts.items()}



	stops = set(stopwords.words("english"))

	def word_shares(row):

		q1_list = str(row['question1']).lower().split()

		q1 = set(q1_list)

		q1words = q1.difference(stops)

		if len(q1words) == 0:

			return '0:0:0:0:0:0:0:0'

        

		q2_list = str(row['question2']).lower().split()

		q2 = set(q2_list)

		q2words = q2.difference(stops)

		if len(q2words) == 0:

			return '0:0:0:0:0:0:0:0'



		words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))



		q1stops = q1.intersection(stops)

		q2stops = q2.intersection(stops)



		q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])

		q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])



		shared_2gram = q1_2gram.intersection(q2_2gram)



		shared_words = q1words.intersection(q2words)

		shared_weights = [weights.get(w, 0) for w in shared_words]

		q1_weights = [weights.get(w, 0) for w in q1words]

		q2_weights = [weights.get(w, 0) for w in q2words]

		total_weights = q1_weights + q1_weights

		

		R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share

		R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)) #count share

		R31 = len(q1stops) / len(q1words) #stops in q1

		R32 = len(q2stops) / len(q2words) #stops in q2

		Rcosine_denominator = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))

		Rcosine = np.dot(shared_weights, shared_weights)/Rcosine_denominator

		if len(q1_2gram) + len(q2_2gram) == 0:

			R2gram = 0

		else:

			R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))

		return '{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R31, R32, R2gram, Rcosine, words_hamming)



	df = pd.concat([df_train, df_test])

	df['word_shares'] = df.apply(word_shares, axis=1, raw=True)



	x = pd.DataFrame()



	x['word_match']       = df['word_shares'].apply(lambda x: float(x.split(':')[0]))

	x['word_match_2root'] = np.sqrt(x['word_match'])

	x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))

	x['shared_count']     = df['word_shares'].apply(lambda x: float(x.split(':')[2]))



	x['stops1_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[3]))

	x['stops2_ratio']     = df['word_shares'].apply(lambda x: float(x.split(':')[4]))

	x['shared_2gram']     = df['word_shares'].apply(lambda x: float(x.split(':')[5]))

	x['cosine']           = df['word_shares'].apply(lambda x: float(x.split(':')[6]))

	x['words_hamming']    = df['word_shares'].apply(lambda x: float(x.split(':')[7]))

	x['diff_stops_r']     = x['stops1_ratio'] - x['stops2_ratio']



	x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))

	x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))

	x['diff_len'] = x['len_q1'] - x['len_q2']

	

	x['caps_count_q1'] = df['question1'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))

	x['caps_count_q2'] = df['question2'].apply(lambda x:sum(1 for i in str(x) if i.isupper()))

	x['diff_caps'] = x['caps_count_q1'] - x['caps_count_q2']



	x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))

	x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))

	x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']



	x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))

	x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))

	x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']



	x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']

	x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']

	x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']



	x['exactly_same'] = (df['question1'] == df['question2']).astype(int)

	x['duplicated'] = df.duplicated(['question1','question2']).astype(int)

	add_word_count(x, df,'how')

	add_word_count(x, df,'what')

	add_word_count(x, df,'which')

	add_word_count(x, df,'who')

	add_word_count(x, df,'where')

	add_word_count(x, df,'when')

	add_word_count(x, df,'why')









	df1 = df_train[['question1']].copy()

	df2 = df_train[['question2']].copy()

	df1_test = df_test[['question1']].copy()

	df2_test = df_test[['question2']].copy()



	df2.rename(columns = {'question2':'question1'},inplace=True)

	df2_test.rename(columns = {'question2':'question1'},inplace=True)



	train_questions = df1.append(df2)

	train_questions = train_questions.append(df1_test)

	train_questions = train_questions.append(df2_test)

	#train_questions.drop_duplicates(subset = ['qid1'],inplace=True)

	train_questions.drop_duplicates(subset = ['question1'],inplace=True)



	train_questions.reset_index(inplace=True,drop=True)

	questions_dict = pd.Series(train_questions.index.values,index=train_questions.question1.values).to_dict()

	train_cp = df_train.copy()

	test_cp = df_test.copy()

	train_cp.drop(['qid1','qid2'],axis=1,inplace=True)



	test_cp['is_duplicate'] = -1

	test_cp.rename(columns={'test_id':'id'},inplace=True)

	comb = pd.concat([train_cp,test_cp])



	x['q1_hash'] = comb['question1'].map(questions_dict)

	x['q2_hash'] = comb['question2'].map(questions_dict)



	q1_vc = x.q1_hash.value_counts().to_dict()

	q2_vc = x.q2_hash.value_counts().to_dict()



	def try_apply_dict(x,dict_to_apply):

		try:

			return dict_to_apply[x]

		except KeyError:

			return 0

	#map to frequency space

	x['q1_freq'] = x['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

	x['q2_freq'] = x['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))

	del q1_vc, q2_vc, comb, test_cp, train_cp, df1, df2, df1_test, df2_test, train_questions, questions_dict

	

	feature_names = list(x.columns.values)

	create_feature_map(feature_names)

	print("Features: {}".format(feature_names))



	x_train = x[:df_train.shape[0]]

	x_test  = x[df_train.shape[0]:]

	y_train = df_train['is_duplicate'].values

	del x, df_train



	if 1: # Now we oversample the negative class - on your own risk of overfitting!

		pos_train = x_train[y_train == 1]

		neg_train = x_train[y_train == 0]



		print("Oversampling started for proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))

		p = 0.165

		scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1

		while scale > 1:

			neg_train = pd.concat([neg_train, neg_train])

			scale -=1

		neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])

		print("Oversampling done, new proportion: {}".format(len(pos_train) / (len(pos_train) + len(neg_train))))



		x_train = pd.concat([pos_train, neg_train])

		y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()

		del pos_train, neg_train

	

	print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))

	clr = train_xgb(x_train, y_train, params)

	del x_train, y_train

	preds = predict_xgb(clr, x_test)



	print("Writing output...")

	sub = pd.DataFrame()

	sub['test_id'] = df_test['test_id']

	sub['is_duplicate'] = preds *.75

	sub.to_csv("xgb_seed{}_n{}.csv".format(RS, ROUNDS), index=False)



	print("Features importances...")

	importance = clr.get_fscore(fmap='xgb.fmap')

	importance = sorted(importance.items(), key=operator.itemgetter(1))

	ft = pd.DataFrame(importance, columns=['feature', 'fscore'])



	ft.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))

	plt.gcf().savefig('features_importance.png')



main()

print("Done.")