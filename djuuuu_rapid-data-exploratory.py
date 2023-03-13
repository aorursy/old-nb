#!/usr/bin/env python
# coding: utf-8



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn
from matplotlib import pyplot as plt
from sklearn import preprocessing as pps
from scipy import sparse
from sklearn.preprocessing import binarize
from sklearn.neighbors import NearestNeighbors
get_ipython().run_line_magic('matplotlib', 'inline')




def clickstream_agg(cstream, user_col, item_col, count_freq = False):
    '''Формируем входные данные для разреженной матрицы
    находим интеракции между пользователями и контентом (бинарные)
    на вход принимаем Series
    '''
    sparse_item_count_index = None
    if count_freq : # возвращаем вектор счётчиков
        cstream = cstream.apply(lambda x: dict(zip(*np.unique(x, return_counts=True))))
        cstream.reset_index(inplace=True, drop=True)
        cstream = cstream.reset_index()
        cstream_item_index = cstream[item_col].apply(lambda x: np.array( list(x.keys())   ) )
        cstream_item_count = cstream[item_col].apply(lambda x: np.array( list(x.values()) ) )
        cstream[item_col] = cstream_item_index
        cstream['item_count'] = cstream_item_count
        sparse_item_count_index = np.concatenate(cstream_item_count.values )
    else:
        cstream = cstream.apply(np.unique)
        # переиндексируем пользователей (label encoding) и приводим к DF
        cstream.reset_index(inplace=True, drop=True)
        cstream = cstream.reset_index()
    cstream_user_index = cstream.apply(lambda x:                                        tuple(np.ones(x[1].shape)*x[0]), axis = 1 )   
    SES_NUM = (cstream[item_col].apply(len)).sum() # количество ненулевых элементов в матрице
    sparse_user_index = np.concatenate(cstream_user_index.values, axis = 0)
    sparse_item_index = np.concatenate(cstream[item_col].values, axis = 0)
    ITEM_NUM = np.unique(sparse_item_index).shape[0]
    USER_NUM =  np.unique(sparse_user_index).shape[0]
    return (ITEM_NUM, USER_NUM, SES_NUM, sparse_user_index, sparse_item_index, sparse_item_count_index, cstream)

def user_interactions(item_num, user_num, interaction_num, sparse_user_index, sparse_item_index, items = None):
    ''' Формируем разреженную матрицу
    
    '''
    fill_values = np.ones( interaction_num ) if items is None else items
    V = sparse.csr_matrix(( fill_values,                      (sparse_user_index, sparse_item_index)),                      shape=(user_num , item_num))
    return V.T




total_orders = pd.read_csv('../input/orders.csv')
train_products = pd.read_csv('../input/order_products__train.csv')
prior_products = pd.read_csv('../input/order_products__prior.csv')
products = pd.read_csv('../input/products.csv')
print (total_orders.head(2))
print (train_products.head(1))
print (products.head(1))




# собираем стрим покупок пользователя (по prior set)
order_stream = total_orders[['user_id', 'order_id', 'order_hour_of_day', 'order_dow']]                                        .merge(prior_products[['order_id',                                                         'product_id']],                                         how='inner', left_on='order_id',                                         right_on='order_id')
# добавляем инфу о категории продукта
raw_stream = order_stream.merge(products[['product_id', 'department_id','product_name']],                                         how='inner', left_on='product_id',                                         right_on='product_id')




# вычисляем среднее число покупок в заказе - сначала считаем число продуктов в каждом заказе
user_metrics = order_stream[['user_id','order_id','product_id']]
user_metrics = user_metrics.groupby(['user_id', 'order_id'])['product_id']
user_metrics = user_metrics.count()
user_metrics = user_metrics.reset_index()
# Теперь считаем среднее число продуктов в заказе для каждого покупателя
user_metrics = user_metrics.groupby(['user_id'])['product_id']
user_metrics = user_metrics.median()
user_metrics = user_metrics.reset_index()
user_metrics.columns = ['user_id',  'check_size']
user_metrics.head(1)




# переиндексируем id департаментов, а то они не с нуля =(
'''
l_department_encoder = pps.LabelEncoder()
raw_stream.department_id = l_department_encoder.fit_transform(raw_stream.department_id.values)
l_product_encoder = pps.LabelEncoder()
raw_stream.product_id = l_product_encoder.fit_transform(raw_stream.product_id.values)
'''




# поменять значение, чтобы посмотреть на графики
has_plot = False




if has_plot:
    raw_stream.groupby('department_id')['user_id'].        agg('count').sort_values(ascending=False).        plot(kind='bar', title = 'Num of users by department')




if has_plot:
    raw_stream.groupby('order_hour_of_day')['order_id'].        agg('count').        plot(kind='bar', title = 'Num of orders per hour')




if has_plot:
    raw_stream.groupby('order_dow')['order_id'].        agg('count').        plot(kind='bar', title = 'Num of orders per day of week')




if has_plot:
    raw_stream.groupby('product_name')['order_id'].agg('count').sort_values(ascending=False)        .head(20).plot(kind='bar', title = 'Freq products')




# формируем историю покупок по каждому пользователю, оставляем только уникальные категории, которые интересуют пользователя
user_hist = raw_stream.groupby('user_id')['department_id'].apply(np.array)
print (user_hist.head(1))
ITEM_NUM, USER_NUM, SESSIONS_NUM, sparse_user_index, sparse_department_index, _, _ =    clickstream_agg(user_hist, user_col="user_id", item_col="department_id")
print ('DEPARTMENT_NUM = {},USER_NUM = {}, SESSIONS_NUM = {}'.format(ITEM_NUM, USER_NUM, SESSIONS_NUM))
V = user_interactions(ITEM_NUM, USER_NUM, SESSIONS_NUM, sparse_user_index, sparse_department_index)
del user_hist




V# Формируем матрицу похожести
xUy = V.dot(V.T) # сколько попокупали одновременно в x и y
# суммируем элементы каждой строки - сколько пользователей покупали в x вместе с другими отделами всего
x = sparse.csr_matrix(V.sum(axis=1))
# умножаем построчно в x не покупали с y, то в total_x будет 0, иначе будет общее число покупателей
total_x = binarize(xUy).multiply(x.T)
# из общего числа пользователей, которые покупали в x вычитаем тех, которые покупали в x и у
_xUy = total_x - xUy
# достаём индексы ненулевых элементов, на которые будем в дальнейшем делить
rows, cols = _xUy.nonzero()
xUy_array = xUy[rows, cols].A.reshape(-1)
# получаем матрицу схожести S
V = sparse.csr_matrix((xUy_array / _xUy.data, (rows, cols)), xUy.shape)
seaborn.heatmap(V.todense())
del V




user_item = raw_stream.groupby('user_id')['product_id'].apply(np.array)
print (user_item.head(2))
ITEM_NUM, USER_NUM, SESSIONS_NUM, sparse_user_index, sparse_department_index, items, stream =    clickstream_agg(user_item, user_col="user_id", item_col="product_id", count_freq = True)
print ('ITEM_NUM = {},USER_NUM = {}, SESSIONS_NUM = {}, num_nonzero = {}'.format(ITEM_NUM, USER_NUM, SESSIONS_NUM, items.shape[0]))
P = user_interactions(ITEM_NUM, USER_NUM, SESSIONS_NUM, sparse_user_index, sparse_department_index, items).T
del user_item
P




def topk_items_matrix(user_ind = 100, k = 10, verbose = False):
    k = int(k)
    # top-k товаров, которые покупали чаще всего
    l = P[user_ind,:].todense().tolist()[0]
    # top-k максимальных элекментов
    top_k_items = (np.argsort(l)[::-1])[:k] # сортировка массива, возвращаем индексы
    if verbose:
        print ('indexes', top_k_items, '\nitem counts', (np.array(l)[[top_k_items]]) )
    #return top_k_items
    return ' '.join([str(x) for x in top_k_items])




def topk_items(user_ind = 100, k = 10, verbose = False):
    k = int(k)
    # top-k товаров, которые покупали чаще всего
    user = stream[stream.index == user_ind]
    l = user.item_count.values[0]
    i = user.product_id.values[0]
    top_k_items = i[np.argsort(l)[::-1][:k]]
    return ' '.join([str(x) for x in top_k_items])
topk_items(100, 10)




# тестовые заказы
test_orders = total_orders[total_orders.eval_set=='test'][['order_id','user_id']].merge(user_metrics,                                         how='left', left_on='user_id',                                         right_on='user_id')
# проверяем, что по всем пользователям нашлись метрики в user_metrics
print (test_orders.check_size.isnull().value_counts())
print (test_orders.head(1))




test_set = test_orders.apply(lambda x: (x[1],int(x[2])) , axis=1).values.tolist()
test_set = pd.DataFrame({'input':test_set})
print (test_set.head(1))
recommend = test_set.input.apply(lambda x: topk_items(*x))
test_orders['recommend'] = recommend
res = test_orders[['order_id','recommend']].sort_values(by='order_id')
res.columns = ['order_id','products']
print (res.head(1))
res.to_csv('./result.csv',sep=',', index = False)




## векторизуем историю покупок
'''
ITEM_NUM, USER_NUM, SESSIONS_NUM, sparse_user_index, sparse_item_index, _ =\
    clickstream_agg(user_item, user_col="user_id", item_col="product_id")
print ('PRODUCT_NUM = {},USER_NUM = {}, SESSIONS_NUM = {}'.format(ITEM_NUM, USER_NUM, SESSIONS_NUM))
U = user_interactions(ITEM_NUM, USER_NUM, SESSIONS_NUM, sparse_user_index, sparse_item_index)
U
'''




'''
nn = NearestNeighbors()
nn.fit(U.T)
'''

