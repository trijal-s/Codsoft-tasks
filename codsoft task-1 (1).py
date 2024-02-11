#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as pl


# In[38]:


df=pd.read_csv("Titanic-Dataset.csv")


# In[39]:


df.head()


# In[40]:


df.describe()


# In[6]:


pip install seaborn


# In[41]:


df["Survived"].value_counts()


# In[16]:


sns.countplot(x=df["Survived"],hue=df['Pclass'])


# In[42]:


sns.countplot(x=df["Sex"],hue=df['Survived'])


# In[19]:


pip install scikit-learn


# In[43]:


from sklearn.preprocessing import LabelEncoder
labenco=LabelEncoder()
df['Sex']=labenco.fit_transform(df['Sex'])
df.head()


# In[23]:


sns.countplot(x=df["Sex"],hue=df['Survived'])


# In[44]:


df.isna().sum()


# In[45]:


df.drop(["Age"],axis=1)


# In[46]:


a=df[["Pclass","Sex"]]
b=df["Survived"]


# In[47]:


from sklearn.model_selection import train_test_split
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2,random_state=0)


# In[48]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(a_train,b_train)


# In[49]:


print(lr.predict(a_test))


# In[50]:


import warnings
warnings.filterwarnings("ignore")
win=lr.predict([[3,1]])
if win==0:
    print("oops!Not survived")
else:
    print("survived")


# In[ ]:




