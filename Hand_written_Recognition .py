#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_openml


# In[2]:


mnist= fetch_openml('mnist_784')


# In[3]:


mnist


# In[4]:


x,y= mnist['data'],mnist['target']


# In[5]:


x.shape


# In[6]:


y.shape


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import matplotlib


# In[8]:


import numpy as np
x=np.array(x)
y=np.array(y)


# In[9]:


some_digit= x[45000]
some_digit_image= some_digit.reshape(28,28)


# In[10]:


plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,interpolation= 'nearest')
plt.axis('off')


# In[11]:


y[45000]


# In[12]:


x_train,x_test= x[:60000],x[60000:]
y_train,y_test= y[:60000],y[60000:]


# In[13]:


shuffle_index = np.random.permutation(60000)
x_train,y_train= x_train[shuffle_index],y_train[shuffle_index]


# ## Creating 2 detector

# In[14]:


y_train=y_train.astype(np.int8)
y_test=y_test.astype(np.int8)
y_train_2= (y_train==2)
y_test_2= (y_test==2)


# In[15]:


y_train_2


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


lr=LogisticRegression()


# In[18]:


lr.fit(x_train,y_train_2)


# In[19]:


lr.predict([some_digit])


# In[20]:


from sklearn.model_selection import cross_val_score
cross_val_score(lr, x_train, y_train_2, cv=3, scoring='accuracy')

