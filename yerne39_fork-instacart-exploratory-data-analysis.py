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

get_ipython().run_line_magic('matplotlib', 'inline')




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

goods.head()




# basic group info (departments)
plt.figure(figsize=(12, 5))
goods.groupby(['department']).count()['product_id'].copy().sort_values(ascending=False).plot(kind='bar', 
                                   #figsize=(12, 5), 
                                   title='Departments: Product #')


# basic group info (top-x aisles)
top_aisles_cnt = 15
plt.figure(figsize=(12, 5))
goods.groupby(['aisle']).count()['product_id'].sort_values(ascending=False)[:top_aisles_cnt].plot(kind='bar', 
                                   #figsize=(12, 5), 
                                   title='Aisles: Product #')

# plot departments volume, split by aisles
f, axarr = plt.subplots(6, 4, figsize=(12, 30))
for i,e in enumerate(departments.department.sort_values(ascending=True)):
    axarr[i//4, i%4].set_title('Dep: {}'.format(e))
    goods[goods.department==e].groupby(['aisle']).count()['product_id']    .sort_values(ascending=False).plot(kind='bar', ax=axarr[i//4, i%4])
f.subplots_adjust(hspace=2)




# load datasets

# train dataset
op_train = pd.read_csv('../input/order_products__train.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
print('Total ordered products(train): {}'.format(op_train.shape[0]))
op_train.head(10)




# test dataset (submission)
test = pd.read_csv('../input/sample_submission.csv', engine='c')
print('Total orders(test): {}'.format(op_train.shape[0]))
test.head()




# prior dataset
op_prior = pd.read_csv('../input/order_products__prior.csv', engine='c', 
                       dtype={'order_id': np.int32, 'product_id': np.int32, 
                              'add_to_cart_order': np.int16, 'reordered': np.int8})
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




get_ipython().run_cell_magic('time', '', "# update by small portions\nfor i in range(len(indexes)-1):\n    order_details = pd.concat(\n        [   \n            order_details,\n            pd.merge(left=pd.merge(\n                            left=op_prior.iloc[indexes[i]:indexes[i+1], :],\n                            right=goods[[\n                                'product_id', \n                                 'aisle_id', \n                                 'department_id'\n                            ]].apply(partial(pd.to_numeric, \n                                             errors='ignore', \n                                             downcast='integer')),\n                            how='left',\n                            on='product_id'\n                            ),\n                     right=orders, \n                     how='left', \n                     on='order_id'\n                ) #.apply(partial(pd.to_numeric, errors='ignore', downcast='integer'))\n        ]\n    )\n        \nprint('Datafame length: {}'.format(order_details.shape[0]))\nprint('Memory consumption: {:.2f} Mb'.format(sum(order_details.memory_usage(index=True, \n                                                                         deep=True) / 2**20)))\n# check dtypes to see if we use memory effectively\nprint(order_details.dtypes)\n\n# make sure we didn't forget to retain test dataset :D\ntest_orders = orders[orders.eval_set == 'test']\n\n# delete (redundant now) dataframes\ndel op_prior, orders")




get_ipython().run_cell_magic('time', '', "# unique orders, product_ordered, users\nprint('Unique users: {}'.format(len(set(order_details.user_id))))\nprint('Unique orders: {}'.format(len(set(order_details.order_id))))\nprint('Unique products bought: {}/{}'.format(len(set(order_details.product_id)), len(goods)))")




# unordered products
unordered = goods[goods.product_id.isin(list(set(goods.product_id) - set(order_details.product_id)))]
print('"Lonesome" products cnt: {}/{}'.format(unordered.shape[0], len(goods)))
unordered.head()




get_ipython().run_cell_magic('time', '', "# popular products (total set, not only train)\ntop = 15\ntop_products = pd.merge(\n    # to see train: \n    # left=pd.DataFrame(order_details[order_details.eval_set == 'train'].groupby(['product_id'])['order_id']\\\n    left=pd.DataFrame(order_details.groupby(['product_id'])['order_id']\\\n    .apply(lambda x: len(x.unique())).sort_values(ascending=False)[:top].reset_index('product_id')),\n    right=goods,\n    how='left')\n\nf, ax = plt.subplots(figsize=(12, 10))\nplt.xticks(rotation='vertical')\nsns.barplot(top_products.product_name, top_products.order_id)\nplt.ylabel('Number of Orders, Containing This Product')\nplt.xlabel('Product Name')")




get_ipython().run_cell_magic('time', '', '# most "frequently" bought products (total set, not only train)\n# most "frequently" ~ time between orders (within selected customer\'s orders), \n# that contain that product, is the least \n#(products, which were bought by more than 100 customers, to omit outliers)\ntop = 15\ncustomer_limit = 100\n\ntemp = order_details.groupby([\'product_id\'])[[\'days_since_prior_order\', \'user_id\']]\\\n.aggregate({\'days_since_prior_order\': np.mean, \'user_id\': len}).reset_index()\n\nfrequent_products = pd.merge(\n    left=pd.DataFrame(temp[temp.user_id > customer_limit].sort_values([\'days_since_prior_order\'], \n                                                                      ascending=True)[:top]),\n    right=goods,\n    how=\'left\')\n\nplt.figure(figsize=(12,6))\nplt.xticks(rotation=\'vertical\')\nsns.barplot(frequent_products.product_name, frequent_products.days_since_prior_order)\nplt.ylabel(\'Average Days Between Orders, Containing This Product\')\nplt.xlabel(\'Product Name\')\n\ndel temp')




get_ipython().run_cell_magic('time', '', 'ord_by_prods = order_details.groupby("order_id")["add_to_cart_order"]\\\n.aggregate(np.max).reset_index()[\'add_to_cart_order\'].value_counts()\n\nprint(\'Most common order contains: {} products\'.format(\n    ord_by_prods[ord_by_prods.values == ord_by_prods.max()].index.values[0]))\n\n# plot it\nplt.figure(figsize=(12, 8))\nplt.xticks(rotation=\'vertical\')\nsns.barplot(ord_by_prods.index, ord_by_prods.values)\nplt.ylabel(\'Number of Orders\')\nplt.xlabel(\'Number of Products in Order\')\nplt.xlim([0, 50])\npass')




get_ipython().run_cell_magic('time', '', "# consider products, purchased in more than X orders\norder_limit = 100\ntop = 15\n\nmo_products = order_details.groupby('product_id')[['reordered', 'order_id']]\\\n.aggregate({'reordered': sum, 'order_id': len}).reset_index()\nmo_products.columns = ['product_id', 'reordered', 'order_cnt']\n\nmo_products['reorder_rate'] = mo_products['reordered'] / mo_products['order_cnt']\nmo_products = mo_products[mo_products.order_cnt > order_limit].sort_values(['reorder_rate'], \n                                                                           ascending=False)[:top]\n\nmo_products = pd.merge(\n    left=mo_products,\n    right=goods,\n    on='product_id')\nmo_products\n\n# plot it\nplt.figure(figsize=(12, 6))\nplt.xticks(rotation='vertical')\nsns.barplot(mo_products.product_name, mo_products.reorder_rate*100)\nplt.ylabel('Reorder Rate, %')\nplt.xlabel('Product Name')\npass")




plt.figure(figsize=(12,6))
order_details.groupby('order_dow')['order_id'].apply(lambda x: len(x.unique())).plot(kind='bar')
plt.xticks(rotation='vertical')
plt.ylabel('Order Count')
plt.xlabel('Day of Week (coded)')
pass




plt.figure(figsize=(12,6))
order_details.groupby('order_hour_of_day')['order_id'].apply(lambda x: 
                                                             len(x.unique())).plot(kind='bar')
plt.xticks(rotation='vertical')
plt.ylabel('Order Count')
plt.xlabel('Hour of a Day (0-23)')
pass




pop_dep = pd.merge(
    left=order_details.groupby('department_id')['order_id'].apply(lambda x: 
                                                                  len(x.unique())).reset_index(),
    right=goods[['department_id', 'department']].drop_duplicates(),
    how='inner',
    on='department_id'
).sort_values(['order_id'], ascending=False)

# plot it
total_orders = len(set(order_details.order_id))

plt.figure(figsize=(12, 6))
plt.xticks(rotation='vertical')
sns.barplot(pop_dep.department, pop_dep.order_id / total_orders * 100)
plt.ylabel('% of Orders, Containing Products from Department, #')
plt.xlabel('Department Name')
pass




get_ipython().run_cell_magic('time', '', "pop_ais = pd.merge(\n    left=order_details.groupby('aisle_id')['order_id'].apply(lambda x: len(x.unique())).reset_index(),\n    right=goods[['aisle_id', 'aisle']].drop_duplicates(),\n    how='inner',\n    on='aisle_id'\n).sort_values(['order_id'], ascending=False)[:top]\n\n# plot it\ntotal_orders = len(set(order_details.order_id))\n\nplt.figure(figsize=(12, 6))\nplt.xticks(rotation='vertical')\nsns.barplot(pop_ais.aisle, pop_ais.order_id / total_orders * 100)\nplt.ylabel('% of Orders, Containing Products from Aisle, #')\nplt.xlabel('Aisle Name')\npass")




get_ipython().run_cell_magic('time', '', 'ocpu = order_details.groupby(\'user_id\')[\'order_id\']\\\n.apply(lambda x: len(x.unique())).reset_index().groupby(\'order_id\').aggregate("count")\n\nprint(\'Most common user made: {} purchases\'.format(\n    ocpu[ocpu.user_id == ocpu.user_id.max()].index.values[0]))\n\nplt.figure(figsize=(12, 6))\nsns.barplot(ocpu.index, ocpu.user_id)\nplt.xticks(rotation=\'vertical\')\nplt.ylabel(\'User Count\')\nplt.xlabel(\'Number of Orders, made by a User\')\npass')




get_ipython().run_cell_magic('time', '', 'dtno = order_details.dropna(axis=0, \n                     subset=[\'days_since_prior_order\']).groupby(\'order_id\')[\'days_since_prior_order\']\\\n.aggregate("mean").reset_index().apply(np.int32).groupby(\'days_since_prior_order\').aggregate("count")\n\nprint(\'Most frequently next orders are made once in: {} days\'.format(\n    dtno[dtno.order_id == dtno.order_id.max()].index.values[0]))\n\nprint(\'We clearly see monthly (>=30) and weekly (7) peaks\')\n\nplt.figure(figsize=(12, 6))\nsns.barplot(dtno.index, dtno.order_id)\nplt.xticks(rotation=\'vertical\')\nplt.ylabel(\'Order Count\')\nplt.xlabel(\'Days Passed Since Last Order\')\npass')




get_ipython().run_cell_magic('time', '', '\npor = order_details.dropna(axis=0, subset=[\'days_since_prior_order\'])\\\n.groupby(\'days_since_prior_order\')[\'reordered\'].aggregate("mean").reset_index()\n\nprint(\'We can see that longer lags leads to lowered probability (new items),\\\n\\nwhile same day orders tends to have more overlapped product list\')\n\nplt.figure(figsize=(12, 6))\nsns.barplot(por.days_since_prior_order, por.reordered*100)\nplt.xticks(rotation=\'vertical\')\nplt.ylabel(\'Probability of Reorder, %\')\nplt.xlabel(\'Days Passed Since Last Order\')\npass')




get_ipython().run_cell_magic('time', '', '\n# share of organic/non-organic products and correspondent orders count\norg = pd.merge(\n    left=order_details[[\'product_id\', \'order_id\']],\n    right=goods[[\'product_id\', \'product_name\']],\n    how=\'left\',\n    on=\'product_id\')\n\norg[\'organic\'] = org.product_name.str.contains(\'organic\').astype(np.int8)\norg = org.groupby(\'order_id\')[\'organic\'].aggregate("max").value_counts()\n\n# plot it\nplt.figure(figsize=(12, 6))\nsns.barplot(org.index, org / org.sum() * 100)\nplt.xticks(rotation=\'vertical\')\nplt.xlabel(\'Order Contains Organics (Boolean)\')\nplt.ylabel(\'% of Orders\')\npass')






