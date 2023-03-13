#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas as pd




train = pd.read_csv('../input/train.tsv', sep='\t')
test = pd.read_csv('../input/test.tsv', sep='\t')
train["source"] = "train"
test["source"] = "test"
train = train.rename(columns={"train_id": "id"})
test = test.rename(columns={"test_id": "id"})
data = pd.concat([train,test])




train.shape,test.shape,data.shape




def split_cat(text):
    try: return text.split("/")
    except: return ("No Label", "No Label", "No Label")




data['general_cat'], data['subcat_1'], data['subcat_2'] = zip(*data['category_name'].apply(lambda x: split_cat(x)))




data.drop('category_name',axis=1,inplace=True)




data.brand_name = data.brand_name.fillna("None")




from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels = ["brand_name","general_cat","subcat_1","subcat_2"]
for label in labels:
    data[label] = data[label].astype(str)
    le.fit(np.hstack(data[label]))
    data[label] = le.transform(data[label])




data.head()




from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.pipeline import FeatureUnion
data.general_cat = data.general_cat.astype(str)
vectorizer = CountVectorizer(token_pattern='\d+')
x = vectorizer.fit_transform(data.general_cat)




data.item_description = data.item_description.fillna("None")
data.item_condition_id = data.item_condition_id.astype(str)
data.shipping = data.shipping.astype(str)
data.general_cat = data.general_cat.astype(str)
data.subcat_1 = data.subcat_1.astype(str)
data.subcat_2 = data.subcat_2.astype(str)
data.brand_name = data.brand_name.astype(str)




data.dtypes




default_preprocessor = CountVectorizer().build_preprocessor()
def preprocessor(field):
    field_idx = list(data.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])


vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor = preprocessor('name'))),
    ('general_cat', CountVectorizer(
        token_pattern='\d+',
        preprocessor=preprocessor('general_cat'))),
    ('subcat_1', CountVectorizer(
        token_pattern='\d+',
        preprocessor=preprocessor('subcat_1'))),
    ('subcat_2', CountVectorizer(
        token_pattern='\d+',
        preprocessor=preprocessor('subcat_2'))),
    ('brand_name', CountVectorizer(
        token_pattern='\d+',
        preprocessor=preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=preprocessor('item_condition_id'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        preprocessor=preprocessor('item_description'))),
])




X = vectorizer.fit_transform(data.values)




trainData = X[:train.shape[0]]
target = np.log1p(train.price)

testData = X[train.shape[0]:]




from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(trainData, target, test_size=0.3, random_state=0)




from sklearn.metrics import mean_squared_error 
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV




ridge_model = Ridge(
    solver='auto', fit_intercept=True, alpha=1.0,
    max_iter=100, normalize=False, tol=0.05, random_state = 1,
)
ridge_model.fit(X_train, y_train)
val_preds= ridge_model.predict(X_valid)
score = mean_squared_error(y_valid, np.array(val_preds))
print("Ridge Score: "+str(np.sqrt(score)))




ridge_modelCV = RidgeCV(
    fit_intercept=True, alphas=[5.0],
    normalize=False, cv = 2, scoring='neg_mean_squared_error',
)
ridge_modelCV.fit(X_train, y_train)
val_preds= ridge_modelCV.predict(X_valid)
score = mean_squared_error(y_valid, np.array(val_preds))
print("RidgeCV Score: "+str(np.sqrt(score)))




preds = ridge_modelCV.predict(testData)

submission= test[["id"]]
submission["price"] = np.expm1(preds)
submission.rename(columns={"id": "test_id"})
submission.to_csv("submission.csv",header=["test_id","price"], index=False)

