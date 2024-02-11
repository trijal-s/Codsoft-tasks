#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib as plt


# In[2]:


sales=pd.read_csv("advertising.csv")


# In[3]:


sales.head()


# In[5]:


sales.shape


# In[6]:


sales.describe()


# In[7]:


sales.info()


# In[8]:


sales.isnull().sum()*100/sales.shape[0]


# In[11]:


sn.boxplot(sales['TV'])


# In[12]:


sn.boxplot(sales['Radio'])


# In[13]:


sn.boxplot(sales['Newspaper'])


# In[14]:


sn.boxplot(sales['Sales'])


# In[16]:


sn.heatmap(sales.corr(), cmap="YlGnBu" ,annot=True)


# In[20]:


a=sales[['TV','Radio','Newspaper']]
b=sales['Sales']


# In[22]:


from sklearn.model_selection import train_test_split
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2)


# In[29]:


from sklearn.linear_model import LinearRegression
log=LinearRegression()


# In[30]:


log.fit(a_train,b_train)


# In[32]:


sa=log.predict(a_test)


# In[ ]:





# In[ ]:




