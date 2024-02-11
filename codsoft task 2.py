#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt


# In[2]:


ir=pd.read_csv("IRIS.csv")


# In[3]:


ir.describe()


# In[4]:


ir.value_counts()


# In[15]:


irs=ir.values
x=irs[:,0:4]
y=irs[:,4]
print(x)


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[25]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)


# In[26]:


predictions=svc.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions)*100)
for i in range(len(predictions)):
    print(y_test[i],predictions[i])


# In[27]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions)*100)


# In[31]:


x_new=np.array([[4,3,2,1],[4.9,2.2,3.8,1.1],[5.3,2.5,4.6,1.9]])
predict=svc.predict(x_new)
print("predictions:{}",format(predict))


# In[ ]:




