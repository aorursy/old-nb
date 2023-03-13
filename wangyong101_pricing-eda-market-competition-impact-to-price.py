#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns

import nltk
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

Sample_Ratio = 0.1
all_cols = ["train_id","brand_name","category_name","item_condition_id","name","shipping","item_description"]
y_col =["price"]
SKIPMAP=False
GLOBAL_SEED=42
get_ipython().system('free -h')




def eda_plot(df,col="",target="log_price",order =None):
    #Usage: eda_plot(df,cols=[],target=""):
    # df: the dataframe
    # cols: the feature columns want to explorate
    # target the target columns want to refer
    if SKIPMAP == False:
        plt.close("all") #clear all plt figure in buffers
           
        if col =="":
            print("No features need print\n")
        else:
            data = df
            if hasattr(data[col], 'cat') or data[col].dtype == np.object:
                fig,ax =plt.subplots(1,2,figsize=(10,1*5),sharey=True)
                sns.countplot(data = data,y=col,ax=ax[0],order=order)
                sns.boxplot(data=data,y=col,x=target,ax=ax[1],orient="h",order=order)
                
            elif (np.issubdtype( data[col].dtype,np.number)):
                #print(data.dtype)
                fig,ax =plt.subplots(1,2,figsize=(10,1*5))
                
                sns.distplot(data[col], ax=ax[0],kde=False)
                sns.boxplot(data=data,x=col,y=target,ax=ax[1])               

            elif data[col].dtype == np.object:
                fig,ax =plt.subplots(1,2,figsize=(10,1*5),sharey=True)

                sns.countplot(data = data,y=col,ax=ax[0],order=order)
                sns.boxplot(data=data,y=col,x=target,ax=ax[1],orient="h",order=order)
            else:
                print(data[col].dtype)
        """
        else:
            fig,ax =plt.subplots(1,2,figsize=(10,1*5))

            for i in range(1):
                col=cols[i]
                data = df[col]

                if (np.issubdtype( data.dtype,np.number)):
                    sns.distplot(data[data.notnull()], ax=ax[0,i])
                    sns.regplot(data,target,ax=ax[1,i])
                    #ax[1,i].set_title(col)

                elif data.dtype== np.object:
                    sns.countplot(y=data[data.notnull()],ax=ax[0,i])
                    sns.boxplot(y=df[col],x=target,ax=ax[1,i])
                    #ax[1,i].set_title(col)
        """
        #plt.subplots_adjust(hspace=0.1,wspace=0.1)




def supply_price_curve(col,train):
    #value_count for all features information. It can help to know how frequency supplied of single name( or categories) under the features
    #log the frequency to capture both high frequency and low frequency.
    tmp = train.groupby(col)[col].transform("count")
    tmp=np.log(tmp)
    if train[col].count() ==0:
        print("{0}-->Empty DataFrame".format(col))
    else:
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        sns.distplot(tmp[tmp.notnull()],kde=False,ax=ax[0])
        sns.regplot(x=tmp,y="log_price",ax=ax[1],data=train)
        ax[0].set_xlabel("Product Qty of each " + col)
        ax[0].set_ylabel("Frequency")
        ax[1].set_xlabel("Product Qty of each " + col)
        plt.subplots_adjust(hspace=0.5,wspace=0.5)
        
    plt.show()




def get_csv(ratio = Sample_Ratio):
    datasets ={"train":"../input/train.tsv",
            "test":"../input/test.tsv"}
    train = pd.DataFrame()
    test = pd.DataFrame()
    for k in datasets.keys():
        file = datasets[k]
        print("\n")
        print("*"*20)
        print("reading {0} data set".format(file))
        data = pd.read_csv(file,delimiter="\t")
        

        data = data.sample(frac=ratio)
        print("----->data set info")
        print(data.info())
        
        missing_sum = data.isnull().sum()
        print("--->missing data")
        print(missing_sum[missing_sum>0])
        
        if k =="train":
            train = data
        elif k == "test":
            test = data
    return train, test
    
train,test = get_csv()




all_cols = ['name', 'item_condition_id', 'category_name', 'brand_name', 'shipping', 'item_description'] 
# exclude the columns of id, and price. ID has different distribution in train &test set. Price is target variance.

def KS_test(test_cols = all_cols,alpha=0.05):
    cols = train[test_cols].select_dtypes(include=[np.number]).columns
    cols_differ = {"ksvalue":[],"pvalue":[]}
    for col in cols:
            pvalue = None
            ksvalue = None
            ksvalue, pvalue = stats.ks_2samp(train[col],test[col])
            cols_differ["ksvalue"].append(ksvalue)
            cols_differ["pvalue"].append(pvalue)

    KStest_df = pd.DataFrame(cols_differ,columns=["ksvalue","pvalue"],index=cols).sort_values(by="pvalue")
    KStest_df["Same_distribution"] = KStest_df.pvalue>alpha
    print(KStest_df["Same_distribution"].head())
KS_test()




train = train.fillna("NA")




train["price"].describe([.25, .5, .75,.99])




fig,ax = plt.subplots(1,2,figsize=(10,5))
sns.distplot(train["price"],ax=ax[0])
sns.distplot(np.log1p(train["price"]),ax=ax[1])
#train["price"].hist(bins=50,ax=ax[0])
#train["price"].apply(np.log1p).hist(bins=50,ax=ax[1])




train["log_price"] = train["price"].apply(np.log1p)
train["log_price"].head()




def eda_zeroprice():
    zero_train = train[train.price==0]
    print("Zero Price item qty {0}".format(zero_train.shape[0]))

    for col in zero_train.columns:
        supply_price_curve(col,train=zero_train)

eda_zeroprice()




train.shape




for col in ["brand_name","category_name","name"]:
    supply_price_curve(col,train =train)




eda_plot(train,"shipping")




eda_plot(train,"item_condition_id")




price_of_brand = train.groupby("brand_name")["price"].agg(["count","mean","std","max","min",])




def top5_plot():
    top5_frequency_brand = train.brand_name.value_counts().head().index

    for col in top5_frequency_brand:
        mask = train.brand_name == col
        train["cat_count"] =train[mask].groupby("category_name")["category_name"].transform("count")
        sns.boxplot(data = train[mask], y = "price",x = "cat_count")
        plt.show()
top5_plot()




def brand_count_price_plot():
    
    brand_count = train["brand_name"].value_counts()
    brand_NA_count =brand_count["NA"]
    print("Total {0} brand in this simpling".format( brand_count.shape[0]))
    print("%{0:.2F} goods with unknow brand name".format(brand_NA_count/brand_count.sum()*100))
    #brand_count.plot.pie()


    
    print(price_of_brand.sort_values(by="count").tail().sort_values(by="count",ascending=False))
    sns.distplot(np.log(brand_count),kde=False,bins=100) #np.log purpose is smooth show all data(without it, the low frequency data will hard to see)
brand_count_price_plot()




def mark_NA_brand():

    mask = train["brand_name"] == "NA"
    train["brand_name_NA"] = 0
    train.loc[mask,"brand_name_NA"] = 1
mark_NA_brand()




def top10_stdev_price_brand():
    
    mask = price_of_brand["count"] >1
    brand_price_top10_std = price_of_brand.loc[mask].sort_values(by="std").tail(10).index
    mask = train["brand_name"].isin( brand_price_top10_std)
    eda_plot(train.loc[mask],"brand_name","price", order =brand_price_top10_std)
top10_stdev_price_brand()




def top10_mean_price_brand():
    brand_price_top10_mean =  price_of_brand.sort_values(by="mean").tail(10).index
    mask = train["brand_name"].isin( brand_price_top10_mean)
    eda_plot(train.loc[mask],"brand_name","price",order=brand_price_top10_mean)
top10_mean_price_brand()




def top10_expensive_brand_price():
    brand_price_top10_max = price_of_brand.sort_values(by="max").tail(10).index
    mask = train["brand_name"].isin( brand_price_top10_max)
    eda_plot(train.loc[mask],"brand_name","price",order=brand_price_top10_max)
top10_expensive_brand_price()




def zero_price_brand():
    Brand_Zero_price= price_of_brand.sort_values(by="min").head(10)
    print(Brand_Zero_price[["min","count"]].sort_values(by="count"))
    brand_price_top10_min =Brand_Zero_price.index
    mask = train["brand_name"].isin( brand_price_top10_min)
    eda_plot(train.loc[mask],"brand_name","price",order=brand_price_top10_min)
zero_price_brand()




def cat_summary():
    print("Total {0} categories was found in this sampling\n".format(train["category_name"].value_counts().shape[0]))
    cat_count = train.category_name.value_counts()
    sns.distplot(cat_count.apply(np.log),kde=False,bins=100)
    print(cat_count.head())

cat_summary()




def brk_categories(df):
    cat_split = df["category_name"].str.split("/",expand=True).    rename(columns={0:"cat_0",1:"cat_1",2:"cat_2",3:"cat_3",4:"cat_4"})
    cat_split = cat_split.apply(lambda x:x.astype("category"))
    if not hasattr(df,"cat_0"):
        df = pd.concat([df,cat_split],axis=1)
    return df

train = brk_categories(train)




tmp = train.cat_1.value_counts()
#sns.distplot(tmp,kde=False,bins=10)
#sns.boxplot("cat_0",y="log_price",data=train,)
eda_plot(df=train, col="cat_0")




for col in ["cat_0","cat_1","cat_2","cat_3"]:
    supply_price_curve(col,train =train)






