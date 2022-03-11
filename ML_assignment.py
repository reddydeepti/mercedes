#!/usr/bin/env python
# coding: utf-8

# In[44]:


import pandas as pd
import numpy as np


# In[45]:


# Step1: Read the train.csv and test.csv data
df_test = pd.read_csv('test.csv')
df_train=pd.read_csv("train.csv")


# In[46]:


df_train.shape


# In[47]:


df_test.shape


# In[48]:


df_train


# In[49]:


df_test


# In[50]:


df_train.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


#extract useful columns for training by removing ID and Y columns
x_train=df_train.iloc[:,2:]
x_test=df_test.iloc[:,1:]
# extract the target column Y
y_train=df_train['y'].values


# In[ ]:





# In[52]:


x_test


# In[53]:


#Check for null in the test and train sets.
x_train.isna().any()


# In[54]:


x_test.isna().any()


# In[ ]:



       


# In[55]:


#Check for unique values for test and train sets and drop them.
for col in x_train.columns:
    car=len(np.unique(x_train[col]))
    if car==1:
        print(col)
        x_train.drop(col, axis=1,inplace=True)
        x_test.drop(col, axis=1, inplace=True)
        


# In[56]:


x_train


# In[57]:


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
      
        
        


# In[58]:


x_train


# In[59]:


x_train.describe()


# In[60]:


x_test.shape


# In[61]:


x_test


# In[62]:


#Perform dimensionality reduction.
from sklearn.decomposition import PCA

sklearn_pca = PCA(n_components=0.95, random_state=420)
sklearn_pca.fit(x_train)


# In[63]:


x_train_transformed=sklearn_pca.transform(x_train)


# In[64]:


print(x_train_transformed.shape)


# In[65]:


print(x_test.shape)


# In[66]:


x_test_transformed=sklearn_pca.transform(x_test)


# In[67]:


print(x_test_transformed.shape)


# In[68]:


print(x_test_transformed)


# In[69]:



from xgboost import XGBRegressor


# In[ ]:





# In[ ]:





# In[ ]:





# In[70]:


# train model using XGBoost
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
model.fit(x_train_transformed, y_train)


# In[71]:


y_pred=model.predict(x_train_transformed)


# In[72]:


from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
print(sqrt(mean_squared_error(y_train, y_pred)))


# In[73]:


#Predict your test_df values using XGBoost
y_test_pred=model.predict(x_test_transformed)


# In[74]:


y_test_pred


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




