#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import lightgbm as lgb
pd.options.display.max_columns = 100

train = pd.read_json('../input/stanford-covid-vaccine/train.json',lines=True)
test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)
sample_sub = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')




import pandas as pd
import numpy as np
import networkx as nx

seq_ls = ['A','U','C','G']
str_ls = ['.','(',')']
loop_ls = ['S','M','I','B','H','E','X']


def series_enc(dic):
    def f(series):
        y = ''
        for s in series:
            y += str(dic[s])
        return y
    return f

bpps_dir = '../input/stanford-covid-vaccine/bpps/'
target_cols = ['reactivity','deg_Mg_pH10','deg_pH10','deg_Mg_50C','deg_50C']
enc = ['sequence','structure','predicted_loop_type']
cnt_feature = ['sequence_cnt0', 'sequence_cnt1', 'sequence_cnt2', 'sequence_cnt3', 'structure_cnt0', 'structure_cnt1', 'structure_cnt2',               'predicted_loop_type_cnt0', 'predicted_loop_type_cnt1', 'predicted_loop_type_cnt2', 'predicted_loop_type_cnt3', 'predicted_loop_type_cnt4',               'predicted_loop_type_cnt5', 'predicted_loop_type_cnt6']
x_col = ['num','bpps_num1','bpps_num2','bpps_num3','between_centers0','closeness_centrality0','eigenvector_centrality_numpy0',         'between_centers1','closeness_centrality1','eigenvector_centrality_numpy1']+enc+cnt_feature

def feature1(df,is_train=True):
    seq_dic = dict(zip(seq_ls,list(range(len(seq_ls)))))
    str_dic = dict(zip(str_ls,list(range(len(str_ls)))))
    loop_dic = dict(zip(loop_ls,list(range(len(loop_ls)))))
    seq_enc = series_enc(seq_dic)
    str_enc = series_enc(str_dic)
    loop_enc = series_enc(loop_dic)
    
    df.loc[:,'sequence_enc'] = df['sequence'].apply(seq_enc)
    df.loc[:,'structure_enc'] = df['structure'].apply(str_enc)
    df.loc[:,'predicted_loop_type_enc'] = df['predicted_loop_type'].apply(loop_enc)
    cnt_feature = []
    for i,s in enumerate(seq_ls):
        df.loc[:,'sequence_cnt'+str(i)]=df['sequence'].apply(lambda x:list(x).count(str(s)))/df.loc[:,'seq_length']
    for i,s in enumerate(str_ls):
        df.loc[:,'structure_cnt'+str(i)]=df['structure'].apply(lambda x:list(x).count(str(s)))/df.loc[:,'seq_length']
    for i,s in enumerate(loop_ls):
        df.loc[:,'predicted_loop_type_cnt'+str(i)]=df['predicted_loop_type'].apply(lambda x:list(x).count(str(s)))/df.loc[:,'seq_length']

    for i in df.index:
        if i%100==0:
            _dfs = make_series1(df.loc[i,:],is_train=is_train)
        else:
            _df = make_series1(df.loc[i,:],is_train=is_train)
            _dfs = pd.concat([_dfs,_df],copy=False)
        if (i%100==99) or i==df.index[-1]:
            if i==99:
                dfs=_dfs
            else:
                dfs=pd.concat([dfs,_dfs],copy=False)
            print(i+1,end=',')
    for c in x_col:
        dfs.loc[:,c] = dfs[c].astype(float)
    return dfs

def make_series1(series,is_train=True):
    num = series['seq_scored']
    df = pd.DataFrame(index=range(num))
    df.loc[:,'id_seqpos'] = df.index
    df.loc[:,'id_seqpos'] = df['id_seqpos'].apply(lambda x:series['id']+'_'+str(x))
    df.loc[:,'id'] = series['id']
    df.loc[:,'num'] = df.index
    for c in cnt_feature:
        df.loc[:,c] = series[c]
    if is_train:
        for c in ['signal_to_noise','SN_filter']:
            df.loc[:,c] = series[c]
    col = ['reactivity','deg_Mg_pH10','deg_pH10','deg_Mg_50C','deg_50C'] +          ['reactivity_error','deg_error_Mg_pH10','deg_error_pH10','deg_error_Mg_50C','deg_error_50C']
    for c in col:
        if is_train:
            df.loc[:,c]=series[c]
    enc = ['sequence','structure','predicted_loop_type']
    for e in enc:
        df.loc[:,e] = list(series[e+'_enc'][:num])
        for n in range(1,6):
            df.loc[:,e+'_n'+str(n)] = list(series[e+'_enc'][n:num+n])
        for m in range(1,6):
            df.loc[:,e+'_m'+str(m)] = [-1]*m+list(series[e+'_enc'][:num-m])
    mat = np.load(bpps_dir+series['id']+'.npy')
    df.loc[:,'bpps_sum'] = mat.sum(axis=0)[:num]/mat.shape[0]
    mat0 = mat>0.0
    df.loc[:,'bpps_num0'] = mat0.sum(axis=0)[:num]/mat0.shape[0]
    mat1 = mat>0.1
    df.loc[:,'bpps_num1'] = mat1.sum(axis=0)[:num]/mat1.shape[0]
    mat2 = mat>0.2
    df.loc[:,'bpps_num2'] = mat2.sum(axis=0)[:num]/mat2.shape[0]
    mat3 = mat>0.3
    df.loc[:,'bpps_num3'] = mat3.sum(axis=0)[:num]/mat3.shape[0]
    
    mat0 = mat0.astype(int)
    for i in range(mat.shape[0]-1):
        mat0[i,i+1]=1
    nodes = np.array(list(range(mat.shape[0])))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    edges = []
    for hi, hv  in enumerate(mat0):
        for wi, wv in enumerate(hv):
            if(wv): edges.append((nodes[hi], nodes[wi]))
    G.add_edges_from(edges)
    df.loc[:,'between_centers0'] = list((nx.betweenness_centrality(G)).values())[:num]
    df.loc[:,'closeness_centrality0'] = list((nx.closeness_centrality(G)).values())[:num]
    df.loc[:,'eigenvector_centrality_numpy0'] = list((nx.eigenvector_centrality_numpy(G)).values())[:num]
    df['clustering0'] = list((nx.clustering(G).values()))[:num]
    
    mat1 = mat1.astype(int)
    for i in range(mat.shape[0]-1):
        mat1[i,i+1]=1
    nodes = np.array(list(range(mat.shape[0])))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    edges = []
    for hi, hv  in enumerate(mat1):
        for wi, wv in enumerate(hv):
            if(wv): edges.append((nodes[hi], nodes[wi]))
    G.add_edges_from(edges)
    df.loc[:,'between_centers1'] = list((nx.betweenness_centrality(G)).values())[:num]
    df.loc[:,'closeness_centrality1'] = list((nx.closeness_centrality(G)).values())[:num]
    df.loc[:,'eigenvector_centrality_numpy1'] = list((nx.eigenvector_centrality_numpy(G)).values())[:num]
    #df['clustering1'] = list((nx.clustering(G).values()))[:num]
    return df




print('calculate train feature')
df = feature1(train.loc[:,:])
print('calculate test feature')
test_df = feature1(test.loc[:,:], is_train=False)

enc = ['sequence','structure','predicted_loop_type']
cnt_feature = ['sequence_cnt0', 'sequence_cnt1', 'sequence_cnt2', 'sequence_cnt3', 'structure_cnt0', 'structure_cnt1', 'structure_cnt2',               'predicted_loop_type_cnt0', 'predicted_loop_type_cnt1', 'predicted_loop_type_cnt2', 'predicted_loop_type_cnt3', 'predicted_loop_type_cnt4',               'predicted_loop_type_cnt5', 'predicted_loop_type_cnt6']
x_col = ['num','bpps_num1','bpps_num2','bpps_num3','between_centers0','closeness_centrality0','eigenvector_centrality_numpy0',         'between_centers1','closeness_centrality1','eigenvector_centrality_numpy1']+enc+cnt_feature
other_col = ['signal_to_noise','SN_filter'] + ['reactivity_error','deg_error_Mg_pH10','deg_error_pH10','deg_error_Mg_50C','deg_error_50C']
for i in range(1,6):
    x_col = x_col+[x+'_n'+str(i) for x in enc]
    x_col = x_col+[x+'_m'+str(i) for x in enc]
for c in x_col:
    df[c] = df[c].astype(float)
    test_df[c] = test_df[c].astype(float)  
df = df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)




params = {
        'task' : 'train',
        'boosting_type' : 'gbdt',
        'objective' : 'regression',
        'metric' : {'l2'},
        'num_leaves' : 101,
        'learning_rate' : 0.005,
        'feature_fraction' : 0.9,
        'bagging_fraction' : 0.8,
        'bagging_freq': 5,
        'verbose' : 0,
        'early_stopping_rounds': 100
}




from sklearn.model_selection import train_test_split, GroupKFold
import lightgbm as lgb
import numpy as np


score = []
E = 10
for e in range(E):
    print(e)
    train_id, val_id = train_test_split(train.loc[train['SN_filter']==1,'id'],train_size=0.8,random_state=e)
    result = df.loc[df['id'].isin(val_id),:].copy()
    for y_col in ['reactivity','deg_Mg_pH10','deg_pH10','deg_Mg_50C','deg_50C']:
        #y_col = 'reactivity'
        print(y_col)
        train_x = df.loc[df['id'].isin(train_id),x_col]
        train_y = df.loc[df['id'].isin(train_id),y_col]
        val_x = df.loc[df['id'].isin(val_id),x_col]
        val_y = df.loc[df['id'].isin(val_id),y_col]


        lgb_train = lgb.Dataset(train_x,label=train_y)
        lgb_val = lgb.Dataset(val_x, label=val_y, reference= lgb_train)
        gbm = lgb.train(params,lgb_train,valid_sets=lgb_val,num_boost_round=100000,verbose_eval=100)
        score.append(gbm.best_score['valid_0']['l2'])
        if e==0:
            test_df[y_col] = gbm.predict(test_df[x_col])/E
        else:
            test_df[y_col] += gbm.predict(test_df[x_col])/E
        #result[y_col] = gbm.predict(val_x)
print(np.mean(score))




sub = sample_sub.copy()[['id_seqpos']]
sub = pd.merge(sub,test_df,on='id_seqpos',how='left').loc[:,['id_seqpos','reactivity','deg_Mg_pH10','deg_pH10','deg_Mg_50C','deg_50C']]
sub.head()




sub = sub.fillna(0)
sub.to_csv('submission.csv',index=False)

