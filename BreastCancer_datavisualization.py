#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


data = pd.read_csv('datasets_180_408_data.csv')


# In[3]:


data.head()


# In[4]:


sns.countplot(x = 'diagnosis', data = data)


# In[5]:


diag = data.diagnosis
list = ['Unnamed: 32','id','diagnosis']
cancer=data.drop(list, axis = 1)
cancer.head()


# In[6]:


cancer.corr()


# In[7]:


cancer[np.logical_and(cancer["area_mean"] > 250, cancer["area_worst"] < 350)]


# In[8]:


f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(cancer.corr(),annot=True , linewidths =.5, fmt = ".1f", ax=ax)
plt.show()


# In[9]:


data_frame=pd.DataFrame(data)
data_frame['diagnosis'].value_counts().plot(kind='bar', alpha = 0.5)
plt.title('Bar plot with Diagnosis Value')
plt.xlabel('diagnosis value')
plt.ylabel('number of counts')
plt.show()


# In[10]:


cancer.plot(kind='scatter', x='smoothness_mean', y='smoothness_worst', grid=True, alpha=0.5, color="blue", figsize=(12, 7))
plt.title("Scatter Plot with Smoothness")
plt.xlabel("smoothness_mean")
plt.ylabel("smoothness_worst")
plt.show()


# In[11]:


cancer.smoothness_mean.plot(kind='line', color='red', alpha=0.5, label="smoothness_mean", grid=True, linestyle=":", figsize=(12,7))
cancer.smoothness_worst.plot(kind='line', color='blue', alpha=0.7, label= "smoothness_worst.plot", grid=True ,linestyle="-.")
plt.title("Line Plot with Smoothness")
plt.legend()
plt.show()


# In[12]:


data_frame['fractal_dimension_mean'].value_counts().plot(kind='hist', color='g', alpha=0.5, figsize=(12,7))
plt.xlabel("Fractal_Dimension_Mean")
plt.show()


# In[ ]:




