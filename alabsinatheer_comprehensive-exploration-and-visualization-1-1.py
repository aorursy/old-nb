#!/usr/bin/env python
# coding: utf-8



# Load packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (10, 6) 




# to customize the displayed area of the dataframe 
pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)




df           = pd.read_csv("../input/train_ver2.csv",
                           dtype={"sexo":str, "ind_nuevo":str, 
                                  "ult_fec_cli_1t":str, 
                                  "indext":str}, nrows=7e6) 

unique_ids   = pd.Series(df["ncodpers"].unique())
unique_id    = unique_ids.sample(n=1.3e5)
df           = df[df.ncodpers.isin(unique_id)]




# Records count
df["ncodpers"].count()




# Change datatype
df["age"]   = pd.to_numeric(df["age"], errors="coerce") 
df["antiguedad"]   = pd.to_numeric(df["antiguedad"], errors="coerce") 
df["indrel_1mes"]   = pd.to_numeric(df["indrel_1mes"], errors="coerce") 




# Check how many missing values in every column
df.isnull().sum()




# Drop the columns with majority of missing values
df = df.drop(["ult_fec_cli_1t", "conyuemp"], axis=1) 




#Impute missing values in the income column 
grouped        = df.groupby("nomprov").agg({"renta":lambda x: x.median(skipna=True)}).reset_index()
new_incomes    = pd.merge(df,grouped,how="inner",on="nomprov").loc[:, ["nomprov","renta_y"]]
new_incomes    = new_incomes.rename(columns={"renta_y":"renta"}).sort_values("renta").sort_values("nomprov")
df.sort_values("nomprov",inplace=True)
df             = df.reset_index()
new_incomes    = new_incomes.reset_index()

df.loc[df.renta.isnull(),"renta"] = new_incomes.loc[df.renta.isnull(),"renta"].reset_index()
df.loc[df.renta.isnull(),"renta"] = df.loc[df.renta.notnull(),"renta"].median()
df.sort_values(by="fecha_dato",inplace=True)




df = df.dropna(axis=0)




df[df["antiguedad"]<0] = 0




# Add a new column of the total number of products per customer per month
df["tot_products"] = df.loc[:,"ind_ahor_fin_ult1":"ind_recibo_ult1"].sum(axis=1)
df["tot_products"]   = pd.to_numeric(df["tot_products"], errors="coerce") 




df['pais_residencia'].describe() 




df = df.loc[lambda df: df.pais_residencia == "ES", :]




df['ind_empleado'].value_counts()




df = df.loc[lambda df: df.ind_empleado == "N", :]




df['age'].hist(bins=50)
plt.title("Customers' Age Distribution")
plt.xlabel("Age(years)")
plt.ylabel("Number of customers") 




# Customers count by channel 
df['canal_entrada'].value_counts().head(15)




df["canal_entrada"].value_counts().plot(x=None, y=None, kind='pie') 




df_a = df.loc[:, ['sexo', 'ind_actividad_cliente']].join(df.loc[:, "ind_ahor_fin_ult1": "ind_recibo_ult1"])
df_a = df_a.groupby(['sexo', 'ind_actividad_cliente']).sum()
df_a = df_a.T




df_a.head()




df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_ncar')
plt.title('Popularity of products by sex and activity index', fontsize=20) 
plt.xlabel('Number of customers', fontsize=17) 
plt.ylabel('Products_names', fontsize=17) 
plt.legend(["Sex:H; Activity_Ind:0", "Sex:H; Activity_Ind:1", "Sex:V; Activity_Ind:0", 
            "Sex:V; Activity_Ind:1"], prop={'size':15}) 




# excluding the dominant product 
exclude = ['ind_cco_fin_ult1']
df_a = df_a.T
df_a = df_a.drop(exclude, axis=1).T




df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_ncar')
plt.title('Popularity of products by sex and activity index', fontsize=20, color='black') 
plt.xlabel('Number of customers', fontsize=17, color='black') 
plt.ylabel('Products_names', fontsize=17, color='black') 
plt.legend(["Sex:H; Activity_Ind:0", "Sex:H; Activity_Ind:1", "Sex:V; Activity_Ind:0", 
            "Sex:V; Activity_Ind:1"], prop={'size':15}) 




df["tot_products"].value_counts()




df_a = df.groupby(['age'])['tot_products'].agg('sum')




df_a.sort_values(ascending=False).head(20)




# Number of products by age 
df_a.plot(kind='bar', colormap='autumn', legend=None) 
plt.xticks(np.arange(0, 120, 10), [str(x) for x in np.arange(0, 120, 10)])
plt.title('Number of products by age') 
plt.xlabel('Age(years)') 
plt.ylabel('Number of products') 




df_a = df.groupby(['segmento'])['tot_products'].agg('sum') 
df_a




df_a = df.groupby(['ind_nuevo'])['tot_products'].agg('count') 
df_a




df_a = df.groupby(['indrel'])['tot_products'].agg('count') 
df_a




df_a = df.groupby(['indrel_1mes'])['tot_products'].agg('count') 
df_a




df_a = df.groupby(['tiprel_1mes'])['tot_products'].agg('count') 
df_a




df_a = df.groupby(['indext'])['tot_products'].agg('count') 
df_a




df_a = (df.groupby(pd.cut(df['renta'], [0,60000,120000,180000,240000, pd.np.inf], right=False))
                     .sum())




df_a["tot_products"]




# Let's drop the columns which are unnecessary for this step
df_a = df_a.loc[:, "ind_ahor_fin_ult1": "ind_recibo_ult1"]
df_a = df_a.T




df_a.head(10)




# Plot of product share for each income group
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_rainbow')
plt.title('Distribution of products among customers by income group', fontsize=20, color='black') 
plt.xlabel('Total number of products', fontsize=17, color='black') 
plt.ylabel('Products_names', fontsize=17, color='black') 
plt.legend(prop={'size':15}, loc=1) 




# exclude the dominant product "ind_cco_fin_ult1"
exclude = ['ind_cco_fin_ult1']
df_a = df_a.T
df_a = df_a.drop(exclude, axis=1).T




df_a.head()




# Plot of product share for each income group; excluding the dominant product 
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_rainbow')
plt.title('Popularity of products by income group', fontsize=20, color='black') 
plt.xlabel('Total number of products', fontsize=17, color='black') 
plt.ylabel('Products_names', fontsize=17, color='black') 
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.1, prop={'size':15}) 




# Let's create age groups
df_a = (df.groupby(pd.cut(df['age'], [0,20,40,60,80,100, pd.np.inf], right=False))
                     .sum())




df_a




# Keep the products columns and discard the others
df_a = df_a.loc[:, "ind_ahor_fin_ult1": "ind_recibo_ult1"]
df_a = df_a.T




df_a.head(10)




# Plot of customers' age distibution of each product 
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='Reds')
plt.title('Customers age distribution of different products', fontsize=20, color='black') 
plt.xlabel('Total number of products', fontsize=17, color='black') 
plt.ylabel('Products_names', fontsize=17, color='black') 
plt.legend(prop={'size':15}, loc=1) 




# exclude the dominant product "ind_cco_fin_ult1"
exclude = ['ind_cco_fin_ult1']
df_a = df_a.T
df_a = df_a.drop(exclude, axis=1).T




# Plot of customers' age distibution of each product (excluding the dominant product) 
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='Blues')
plt.title('Customers age distribution of different products', fontsize=20, color='black') 
plt.xlabel('Total number of products', fontsize=17, color='black') 
plt.ylabel('Products_names', fontsize=17, color='black') 
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.1, prop={'size':15}) 




df["canal_entrada"].value_counts().head(10)




# Let's extract the necessary columns for this step
df_a = df.loc[:, ['canal_entrada']].join(df.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'])




# Let's subset the data to keep only the records from the five major channels
subset = ["KHE", "KAT", "KFC", "KFA", "KHK"]
df_a = df_a.loc[df_a['canal_entrada'].isin(subset)]




df_a = df_a.groupby("canal_entrada").agg("sum")
df_a = df_a.T




# Channels used by the customer to join and the purchased products
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_rainbow')
plt.title('Channels used by the customers to join and associated product uses', fontsize=20, color='black') 
plt.xlabel('Total number of customers', fontsize=17, color='black') 
plt.ylabel('Products names', fontsize=17, color='black') 
plt.legend(prop={'size':15}) 




# exclude the dominant product "ind_cco_fin_ult1"
exclude = ['ind_cco_fin_ult1']
df_a = df_a.T
df_a = df_a.drop(exclude, axis=1).T




# Channels share distribution of each product, excluding the dominant product
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_rainbow')
plt.title('Channels used to join for each product', fontsize=20, color='black') 
plt.xlabel('Total number of customers', fontsize=17, color='black') 
plt.ylabel('Products_names', fontsize=17, color='black') 
plt.legend(["KAT", "KFA", "KFC", "KHE", "KHK"], prop={'size':15}, loc=1) 
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.1, prop={'size':15}) 




# Let's extract the necessary columns for this step
df_a = df.loc[:, ['antiguedad']].join(df.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'])




# Let's create seniority groups
df_a = (df_a.groupby(pd.cut(df_a['antiguedad'], [0,50,100,150,200, pd.np.inf], right=False))
                     .sum())




df_a.head()




exclude = ["antiguedad"]
df_a = df_a.drop(exclude, axis=1).T




# Customers' seniority distribution of each product
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_rainbow')
plt.title('Customers seniority distribution of each product', fontsize=20, color='black') 
plt.xlabel('Customer seniority', fontsize=17, color='black') 
plt.ylabel('Product names', fontsize=17, color='black') 
plt.legend([[0, 50], [50, 100], [100, 150], [150, 200], [200, inf]], prop={'size':15}) 




# exclude the dominant product "ind_cco_fin_ult1"
exclude = ['ind_cco_fin_ult1']
df_a = df_a.T
df_a = df_a.drop(exclude, axis=1).T




# Customers' seniority distribution of each product
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_rainbow')
plt.title('Customers seniority distribution of each product', fontsize=20, color='black') 
plt.xlabel('Customer seniority', fontsize=17, color='black') 
plt.ylabel('Product names', fontsize=17, color='black') 
plt.legend([[0, 50], [50, 100], [100, 150], [150, 200], [200, inf]], prop={'size':15}) 
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.1, prop={'size':15}) 




# Extract total number of products 
df_a = df["tot_products"].value_counts()
df_a = pd.DataFrame(df_a)




df["ncodpers"].count()




# calculate the percentage of customers using different number of products
count = df["ncodpers"].count()
df_a["percentage"] = (df_a["tot_products"]/count)*100
df_a




# extract those customers who purchased only one product had being current account ("ind_cco_fin_ult1") 
df_a = df[df["tot_products"]==1]  
df_a = df_a[df_a["ind_cco_fin_ult1"]==1]  




a = df_a["ncodpers"].count() # Observations where customers had only one product being the current account 
b = len(df) # Total number of observations
c = len(df[df["tot_products"]==1]) # Observations where customers had only one account

print("%.2f" % ((c/b)*100), "% of the customers had purchased only one product") 
print("%.2f" % ((a/b)*100), "% of the customers had the current account as the only one product") 
print("%.2f" % ((47.12/55.74)*100), "% of the customers when they have only one product, this product is the current account")




print("%.2f" % (55.74 - 47.12 ), "% of the customers have only one account being not the current account") 




# extract the necessary columns
df_a = df[df["tot_products"]==1]  # cases where the total products is one
df_a = df_a.loc[:, ["tot_products"]].join(df_a.loc[:, "ind_ahor_fin_ult1":"ind_recibo_ult1"]) 




df_a = df_a.groupby("tot_products").agg("sum")
df_a = df_a.T




df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='winter')
plt.title('Number of times each product was chosen as the only product in case of the total products is one', fontsize=20) 
plt.xlabel('times of occurences', fontsize=17, color='black') 
plt.ylabel('products names', fontsize=17, color='black') 




# extract those customers who purchased two products with current account being one of them
df_a = df[df["tot_products"]==2]  
df_a = df_a[df_a["ind_cco_fin_ult1"]==1]  




df_a["ncodpers"].count()




# extract the necessary columns
df_a = df_a.loc[:, ["tot_products"]].join(df_a.loc[:, "ind_ahor_fin_ult1":"ind_recibo_ult1"]) 




df_a = df_a.groupby("tot_products").agg("sum")
df_a = df_a.T




df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='winter')
plt.title('Number of times each product was chosen along with the dominant product in case of the total products is two', fontsize=20) 
plt.xlabel('times of occurences', fontsize=17, color='black') 
plt.ylabel('products names', fontsize=17, color='black') 




# extract those customers who did not purchase any products in any month
df_a = df[df["tot_products"]==0]




df_a["ind_actividad_cliente"].value_counts() 




# extract those customers who purchased three products in any single month
df_a = df[df["tot_products"]==3]  




df_a = df_a.loc[:, ["tot_products"]].join(df_a.loc[:, "ind_ahor_fin_ult1":"ind_recibo_ult1"]) 




df_a = df_a.groupby("tot_products").agg("sum")
df_a = df_a.T




df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='winter')
plt.title('Number of times each product was chosen in case of the total products is three in any month', fontsize=20, color='black') 
plt.xlabel('times of occurences', fontsize=17, color='black') 
plt.ylabel('products names', fontsize=17, color='black') 




# Categorize by total products
df_a = df.loc[:, ["tot_products"]].join(df.loc[:, "ind_ahor_fin_ult1":"ind_recibo_ult1"]) 
df_a = df_a.groupby("tot_products").agg("sum")
df_a = df_a.T




df_a.head()




# percentage of each product contribution of those customers who only purchased one product in any month
a = df_a[1]
b = df_a[1].sum()
c = (a/b)*100
c = c.sort_values(ascending=False)
c




c[0]




print("Wow, about", "%.2f" % (c[0]), "% of the customers purchase only the current account in case they purchase only one product.") 




# percentage of each product contribution of those customers who purchased two products in any month
a = df_a[2]
b = df_a[2].sum()
c = (a/b)*100
c = c.sort_values(ascending=False)
c




print("In case of two products purchased by the customer, about", "%.2f" % (c[0]), "% of the customers have got a (current account) and in", "%.2f" %(c[1]),"% of the cases it is combined with (Particular Account) and in", "%.2f"% (c[2]),"% of the cases it is combined with (Direct Debit)") 




# percentage of each product contribution of those customers who purchased three products in any month
a = df_a[3]
b = df_a[3].sum()
c = (a/b)*100
c.sort_values(ascending=False)




df_a = df.loc[:, ['age']].join(df.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'])




df_a = (df_a.groupby(pd.cut(df_a['age'], [0,18,25,35,45,55, pd.np.inf], right=False))
                     .sum())




df_a.head()




del df_a["age"]




df_a = df_a.T




# Customers age distribution of each product
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_rainbow')
plt.title('Customers age distribution of each product', fontsize=20, color='black') 
plt.xlabel('Total number of customers', fontsize=17, color='black') 
plt.ylabel('Products names', fontsize=17, color='black') 
plt.legend([[0, 18], [18, 25], [25, 35], [35, 45], [45, 55], [55, inf]], prop={'size':15}) 




df_a = df.loc[:,["age"]].join(df.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'])




fnc = {c:'sum' for c in df_a.columns.drop(['age']).tolist()}
fnc['age'] = 'mean'




df_a = df_a.groupby('age').agg(fnc).reindex_axis(df_a.columns.drop('age'), 1) 




df_a.head()




# Products distribution by age
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,42], colormap='hsv')
plt.title('Products distribution by age', fontsize=20, color='black') 
plt.xlabel('Total number of customers', fontsize=17, color='black') 
plt.ylabel('Age (years)', fontsize=17, color='black') 
plt.legend(prop={'size':15}) 




df_a = df.loc[:, ['segmento']].join(df.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'])




df_a = df_a.groupby("segmento").agg("sum")
df_a = df_a.T




df_a.head()




# Customers segment of each product
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='gist_rainbow')
plt.title('Customers segmentation of products', fontsize=20, color='black') 
plt.xlabel('Total number of customers', fontsize=17, color='black') 
plt.ylabel('Products names', fontsize=17, color='black') 
plt.legend(prop={'size':15}) 




# Let's extract the necessary columns for this step
df_a = df.loc[:, ['ind_actividad_cliente']].join(df.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'])




df_a = df_a.groupby("ind_actividad_cliente").agg("sum")
df_a = df_a.T




# Purchased products types by customer activity index
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='prism')
plt.title('Purchased products types by customer activity index ', fontsize=20, color='black') 
plt.xlabel('Total number of customers', fontsize=17, color='black') 
plt.ylabel('Products names', fontsize=17, color='black') 
plt.legend(["Inactive", "Active"], prop={'size':15}) 




# Let's extract the necessary columns for this step
df_a= df.loc[:, ['sexo']].join(df.loc[:, 'ind_ahor_fin_ult1':'ind_recibo_ult1'])




df_a = df_a.groupby("sexo").agg("sum")
df_a = df_a.T




# Percentage of purchased products by sex
df_a.plot(kind='barh', stacked=True, fontsize=14, figsize=[16,12], colormap='prism')
plt.title('Purchased products by sex ', fontsize=20, color='black') 
plt.xlabel('Total number of customers', fontsize=17, color='black') 
plt.ylabel('Products names', fontsize=17, color='black') 
plt.legend(prop={'size':15}) 




df["sexo"].value_counts()




a = df["ncodpers"][df["sexo"]=="H"].count()
b = df["ncodpers"][df["sexo"]=="V"].count()
(a/b)*100




df_a = df.loc[:, ['age', 'renta', 'antiguedad']].join(df.loc[:, 'ind_ahor_fin_ult1':'tot_products'])




df_a.head()




df_a = df_a.dropna(axis=0)




df_a = df_a.groupby("tot_products").agg("mean")




df_b = df_a.loc[:, ['age', 'renta', 'antiguedad']]




df_b.head()




df_a = df_b["antiguedad"][0:10]




# Total number of products by seniority
df_a.plot(kind='bar', fontsize=14, figsize=[16,12], colormap='prism')
plt.title('Total number of products by seniority', fontsize=20, color='black') 
plt.xlabel('Total number of products', fontsize=17, color='black') 
plt.ylabel('Average seniority', fontsize=17, color='black') 
#plt.legend(prop={'size':15}) 




df_a = df_b["age"]




# Total number of products by age
df_a.plot(kind='bar', fontsize=14, figsize=[16,12], colormap='prism')
plt.title('Total number of products by age', fontsize=20, color='black') 
plt.xlabel('Total number of products', fontsize=17, color='black') 
plt.ylabel('Average age', fontsize=17, color='black') 
plt.legend(prop={'size':15}) 




df_a = df_b["renta"]




# Total number of products by income
df_a.plot(kind='bar', fontsize=14, figsize=[16,12], colormap='prism')
plt.title('Total number of products by income', fontsize=20, color='black') 
plt.xlabel('Total number of products', fontsize=17, color='black') 
plt.ylabel('Average income', fontsize=17, color='black') 
plt.legend(prop={'size':15}) 

