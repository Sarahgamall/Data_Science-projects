#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised Machine Learning
# 
# predict the percentage of a student based on the number of hours 

# 1- Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 2- read the data

# In[2]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"

dataset = pd.read_csv(url)


# In[3]:


dataset.shape


# In[4]:


dataset.head(10)


# In[5]:


dataset.describe()


# 3- Visualising data

# In[7]:


dataset.plot(x='Hours',y='Scores',style = 'o')
plt.title('precentage scores vs Hours of study')
plt.xlabel('Hours of study')
plt.ylabel('precentage score')
plt.show()


# In[29]:


X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:,1].values


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("the model is trained")


# In[35]:


line = regressor.coef_*X+regressor.intercept_
print(regressor.coef_)
print(regressor.intercept_)


# 4- Making Predictions

# In[36]:


y_pred = regressor.predict(X_test)


# In[37]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[38]:


plt.scatter(X, y)
plt.plot(X, line);
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.show()


# In[ ]:




