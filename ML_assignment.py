#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

# Step1: Read the train.csv and test.csv data
df_test = pd.read_csv('test.csv')
df_train=pd.read_csv("train.csv")
df_train.shape
df_test.shape
df_train
df_test
df_train.describe()


#extract useful columns for training by removing ID and Y columns
x_train=df_train.iloc[:,2:]
x_test=df_test.iloc[:,1:]
# extract the target column Y
y_train=df_train['y'].values
x_test

#Check for null in the test and train sets.
x_train.isna().any()
x_test.isna().any()
#Check for unique values for test and train sets and drop them.
for col in x_train.columns:
    car=len(np.unique(x_train[col]))
    if car==1:
        print(col)
        x_train.drop(col, axis=1,inplace=True)
        x_test.drop(col, axis=1, inplace=True)
        
x_train

# Import label encoder
from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
 
for col in x_train.columns:
    typ = df_train[col].dtype
    if typ == 'object':
        # Encode labels in column 'species'.
        x_train[col]= label_encoder.fit_transform(x_train[col])
        x_test[col]= label_encoder.fit_transform(x_test[col])
      
x_train
x_train.describe()
x_test.shape
x_test

#Perform dimensionality reduction.
from sklearn.decomposition import PCA

sklearn_pca = PCA(n_components=0.95, random_state=420)
sklearn_pca.fit(x_train)

x_train_transformed=sklearn_pca.transform(x_train)
print(x_train_transformed.shape)
print(x_test.shape)
x_test_transformed=sklearn_pca.transform(x_test)
print(x_test_transformed.shape)
print(x_test_transformed)

# train model using XGBoost
from xgboost import XGBRegressor
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(x_train_transformed, y_train)

y_pred=model.predict(x_train_transformed)

from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
print(sqrt(mean_squared_error(y_train, y_pred)))


#Predict your test_df values using XGBoost
y_test_pred=model.predict(x_test_transformed)
y_test_pred

