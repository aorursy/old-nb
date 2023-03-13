#!/usr/bin/env python
# coding: utf-8



import pandas as pd # dataframes
import numpy as np # algebra & calculus
import nltk # text preprocessing & manipulation
# from textblob import TextBlob
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting

from functools import partial # to reduce df memory consumption by applying to_numeric

color = sns.color_palette() # adjusting plotting style
import warnings
warnings.filterwarnings('ignore') # silence annoying warnings




# aisles
aisles = pd.read_csv('../input/aisles.csv', engine='c')
print('Total aisles: {}'.format(aisles.shape[0]))
aisles.head()




# departments
departments = pd.read_csv('../input/departments.csv', engine='c')
print('Total departments: {}'.format(departments.shape[0]))
departments.head()




# products
products = pd.read_csv('../input/products.csv', engine='c')
print('Total products: {}'.format(products.shape[0]))
products.head(5)




# combine aisles, departments and products (left joined to products)
goods = pd.merge(left=pd.merge(left=products, right=departments, how='left'), right=aisles, how='left')
# to retain '-' and make product names more "standard"
goods.product_name = goods.product_name.str.replace(' ', '_').str.lower() 
print(goods.info())

goods.head()




# load datasets

# train dataset
op_train = pd.read_csv('../input/order_products__train.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
print('Total ordered products(train): {}'.format(op_train.shape[0]))
op_train.head(10)




# test dataset (submission)
test = pd.read_csv('../input/sample_submission.csv', engine='c')
print('Total orders(test): {}'.format(test.shape[0]))
test.head()




#prior dataset
op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c', 
                       dtype={'order_id': np.int32, 
                              'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 
                              'reordered': np.int8})

print('Total ordered products(prior): {}'.format(op_prior.shape[0]))
op_prior.head()




# orders
orders = pd.read_csv('../input/orders.csv', engine='c', dtype={'order_id': np.int32, 
                                                           'user_id': np.int32, 
                                                           'order_number': np.int32, 
                                                           'order_dow': np.int8, 
                                                           'order_hour_of_day': np.int8, 
                                                           'days_since_prior_order': np.float16})
print('Total orders: {}'.format(orders.shape[0]))
print(orders.info())
orders.head()




from functools import partial

# merge train and prior together iteratively, to fit into 8GB kernel RAM
# split df indexes into parts
indexes = np.linspace(0, len(op_prior), num=10, dtype=np.int32)

# initialize it with train dataset
order_details = pd.merge(
                left=op_train,
                 right=orders, 
                 how='left', 
                 on='order_id'
        ).apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))

# add order hierarchy
order_details = pd.merge(
                left=order_details,
                right=goods[['product_id', 
                             'aisle_id', 
                             'department_id']].apply(partial(pd.to_numeric, 
                                                             errors='ignore', 
                                                             downcast='integer')),
                how='left',
                on='product_id'
)

print(order_details.shape, op_train.shape)

# delete (redundant now) dataframes
del op_train

order_details.head()




get_ipython().run_cell_magic('time', '', "# update by small portions\nfor i in range(len(indexes)-1):\n    order_details = pd.concat(\n        [   \n            order_details,\n            pd.merge(left=pd.merge(\n                            left=op_prior.loc[indexes[i]:indexes[i+1], :],\n                            right=goods[['product_id', \n                                         'aisle_id', \n                                         'department_id' ]].apply(partial(pd.to_numeric, \n                                                                          errors='ignore', \n                                                                          downcast='integer')),\n                            how='left',\n                            on='product_id'\n                            ),\n                     right=orders, \n                     how='left', \n                     on='order_id'\n                ) #.apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))\n        ]\n    )\n        \nprint('Datafame length: {}'.format(order_details.shape[0]))\nprint('Memory consumption: {:.2f} Mb'.format(sum(order_details.memory_usage(index=True, \n                                                                         deep=True) / 2**20)))\n# check dtypes to see if we use memory effectively\nprint(order_details.dtypes)\n\n# make sure we didn't forget to retain test dataset :D\ntest_orders = orders[orders.eval_set == 'test']\n\n# delete (redundant now) dataframes\ndel op_prior, orders")




get_ipython().run_cell_magic('time', '', "# dumb submission\ntest_history = order_details[(order_details.user_id.isin(test_orders.user_id))]\\\n.groupby('user_id')['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index()\ntest_history.columns = ['user_id', 'products']\n\ntest_history = pd.merge(left=test_history, \n                        right=test_orders, \n                        how='right', \n                        on='user_id')[['order_id', 'products']]\n\ntest_history.to_csv('dumb_submission.csv', encoding='utf-8', index=False)")




get_ipython().run_cell_magic('time', '', "# dumb submission\ntest_history = order_details[(order_details.user_id.isin(test_orders.user_id)) \n                             & (order_details.reordered == 1)]\\\n.groupby('user_id')['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index()\ntest_history.columns = ['user_id', 'products']\n\ntest_history = pd.merge(left=test_history, \n                        right=test_orders, \n                        how='right', \n                        on='user_id')[['order_id', 'products']]\n\ntest_history.to_csv('dumb2_subm.csv', encoding='utf-8', index=False)")




get_ipython().run_cell_magic('time', '', "test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]\nlast_orders = test_history.groupby('user_id')['order_number'].max()\n\ndef get_last_orders():\n    t = pd.merge(\n            left=pd.merge(\n                    left=last_orders.reset_index(),\n                    right=test_history,\n                    how='inner',\n                    on=['user_id', 'order_number']\n                )[['user_id', 'product_id']],\n            right=test_orders[['user_id', 'order_id']],\n            how='left',\n            on='user_id'\n        ).groupby('order_id')['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index()\n    t.columns = ['order_id', 'products']\n    return t\n\n# save submission\nget_last_orders().to_csv('less_dumb_subm_last_order.csv', encoding='utf-8', index=False)")




get_ipython().run_cell_magic('time', '', "test_history = order_details[(order_details.user_id.isin(test_orders.user_id))]\nlast_orders = test_history.groupby('user_id')['order_number'].max()\n\ndef get_last_orders_reordered():\n    t = pd.merge(\n            left=pd.merge(\n                    left=last_orders.reset_index(),\n                    right=test_history[test_history.reordered == 1],\n                    how='left',\n                    on=['user_id', 'order_number']\n                )[['user_id', 'product_id']],\n            right=test_orders[['user_id', 'order_id']],\n            how='left',\n            on='user_id'\n        ).fillna(-1).groupby('order_id')['product_id'].apply(lambda x: ' '.join([str(int(e)) for e in set(x)]) \n                                                  ).reset_index().replace(to_replace='-1', \n                                                                          value='None')\n    t.columns = ['order_id', 'products']\n    return t\n\n# save submission\nget_last_orders_reordered().to_csv('less_dumb_subm_last_order_reordered_only.csv', \n                         encoding='utf-8', \n                         index=False)")

