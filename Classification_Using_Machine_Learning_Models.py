#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # Read The Data

# In[2]:


features =  ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "MartialStatus", "Occupation", "Relationship", "Race", "Sex", "CapitalGain", "CapitalLoss", "Hoursperweek", "Country", "Target"]


# In[3]:


data_train = pd.read_csv('adult.data', names = features)
data_train.head()


# In[4]:


data_test = pd.read_csv('adult.test', names = features)
data_test.head()


# # Data Pre-Processing

# In[5]:


data_train.shape


# In[6]:


data_test.shape


# In[7]:


data_train.info()


# In[8]:


data_test.info()


# In[9]:


data_train.isnull().sum()


# In[10]:


data_test.isnull().sum()


# # Filling the NAN Values

# In[11]:


data_test['Workclass'].fillna('Workclass', inplace = True)


# In[12]:


data_test['fnlwgt'].fillna('fnlwgt', inplace = True)


# In[13]:


data_test['Education'].fillna('Education', inplace = True)


# In[14]:


data_test['Education-Num'].fillna('Education-Num', inplace = True)


# In[15]:


data_test['MartialStatus'].fillna('MartialStatus', inplace = True)


# In[16]:


data_test['Occupation'].fillna('Occupation', inplace = True)


# In[17]:


data_test['Relationship'].fillna('Relationship', inplace = True)


# In[18]:


data_test['Race'].fillna('Race', inplace = True)


# In[19]:


data_test['Sex'].fillna('Sex', inplace = True)


# In[20]:


data_test['CapitalGain'].fillna('CapitalGain', inplace = True)


# In[21]:


data_test['CapitalLoss'].fillna('CapitalLoss', inplace = True)


# In[22]:


data_test['Hoursperweek'].fillna('Hoursperweek', inplace = True)


# In[23]:


data_test['Country'].fillna('Country', inplace = True)


# In[24]:


data_test['Target'].fillna('Target', inplace = True)


# # Label Encoder 

# In[25]:


from sklearn.preprocessing import LabelEncoder


# In[26]:


le = LabelEncoder()


# # For Train Data

# In[27]:


label = data_train.select_dtypes(include = 'object').columns

for i in label:
    
    data_train[i] = le.fit_transform(data_train[i].astype(str))


# In[28]:


data_train.info()


# In[29]:


data_train.isnull().sum()


# # For Test Data

# In[30]:


label1 = data_test.select_dtypes(include = 'object').columns

for i in label1:
    
    data_test[i] = le.fit_transform(data_test[i].astype(str))


# In[31]:


data_test.info()


# In[32]:


data_test.isnull().sum()


# # Seperation of X and Y Lables of Train Data

# In[34]:


X_train = data_train.iloc[:,0:14]
Y_train = data_train.iloc[:,-1]


# # Seperation of X and Y Lables of Test Data

# In[35]:


X_test = data_test.iloc[:,0:14]
Y_test = data_test.iloc[:,-1]


# # Naive Bayes Model

# In[36]:


from sklearn import naive_bayes


# In[37]:


Naive_Bayes = naive_bayes.MultinomialNB()


# In[38]:


from sklearn.metrics import accuracy_score


# In[39]:


Naive_Bayes.fit(X_train, Y_train)
Y_pred = Naive_Bayes.predict(X_test)
Accuracy_of_Naive_Bayes = Naive_Bayes.score(X_test, Y_test)*100
print ("Accuracy of Naive Bayes Model:", Accuracy_of_Naive_Bayes)


# In[40]:


from sklearn.metrics import recall_score


# In[41]:


Recall_Score_of_Naive_Bayes = recall_score(Y_test, Y_pred, average='macro') * 100
print ("Recall Score of Naive Bayes Model:", Recall_Score_of_Naive_Bayes)


# In[42]:


from sklearn.metrics import precision_score


# In[43]:


Precision_Score_of_Naive_Bayes = precision_score(Y_test, Y_pred, average = 'macro') * 100
print ("Precision Score of Naive Bayes Model:", Precision_Score_of_Naive_Bayes)


# In[44]:


from sklearn.metrics import f1_score


# In[45]:


F1_Score_of_Naive_Bayes = f1_score(Y_test, Y_pred, average='macro') * 100
print ("F1 Score of Naive Bayes Model:", F1_Score_of_Naive_Bayes)


# # KNN Model

# In[46]:


from sklearn.neighbors import KNeighborsClassifier


# In[47]:


KNN = KNeighborsClassifier(n_neighbors = 5)


# In[48]:


from sklearn.metrics import accuracy_score


# In[49]:


KNN.fit(X_train, Y_train)
Y_pred = KNN.predict(X_test)
Accuracy_of_KNN = (accuracy_score(Y_pred, Y_test)) * 100
print ("Accuracy of KNN Model:", Accuracy_of_KNN)


# In[50]:


from sklearn.metrics import recall_score


# In[51]:


Recall_Score_of_KNN = recall_score(Y_test, Y_pred, average='macro')*100
print ("Recall Score of KNN Model:", Recall_Score_of_KNN)


# In[52]:


from sklearn.metrics import precision_score


# In[53]:


Precision_Score_of_KNN = precision_score(Y_test, Y_pred, average = 'macro') * 100
print ("Precision Score of KNN Model:", Precision_Score_of_KNN)


# In[54]:


from sklearn.metrics import f1_score


# In[55]:


F1_Score_of_KNN = f1_score(Y_test, Y_pred, average='macro') * 100
print ("F1 Score of KNN Model:", F1_Score_of_KNN)


# # Comparsion of Both KNN And Naive Bayes Model

# In[56]:


print("Accuray of KNN: ", Accuracy_of_KNN)
print("Accuray of Naive Bayes: ", Accuracy_of_Naive_Bayes)


# In[57]:


x = ['KNN','Naive Bayes']
y = [Accuracy_of_KNN, Accuracy_of_Naive_Bayes]
y


# In[58]:


plt.figure(figsize = (8,8))
plt.bar(x,y, color = ['yellow', 'skyblue'])
plt.xlabel('Models')
plt.ylabel("Accuracy")
plt.title("Accuracy of KNN Model and Naive Bayes")
plt.show()


# In[59]:


print ("Recall Score of Naive Bayes Model:", Recall_Score_of_Naive_Bayes)
print ("Recall Score of Naive Bayes Model:", Recall_Score_of_KNN)


# In[60]:


x = ['KNN','Naive Bayes']
y = [Recall_Score_of_KNN, Recall_Score_of_Naive_Bayes]
y


# In[61]:


plt.figure(figsize = (8,8))
plt.bar(x,y, color = ['red', 'orange'])
plt.xlabel('Models')
plt.ylabel("Recall Score")
plt.title("Recall Score of KNN Model and Naive Bayes")
plt.show()


# In[62]:


print ("Precision Score of Naive Bayes Model:", Precision_Score_of_Naive_Bayes)
print ("Precision Score of Naive Bayes Model:", Precision_Score_of_KNN)


# In[63]:


x = ['KNN','Naive Bayes']
y = [Precision_Score_of_KNN, Precision_Score_of_Naive_Bayes]
y


# In[64]:


plt.figure(figsize = (8,8))
plt.bar(x,y, color = ['green', 'purple'])
plt.xlabel('Models')
plt.ylabel("Precision Score")
plt.title("Precision Score of KNN Model and Naive Bayes")
plt.show()


# In[65]:


print ("F1 Score of Naive Bayes Model:", F1_Score_of_Naive_Bayes)
print ("F1 Score of Naive Bayes Model:", F1_Score_of_KNN)


# In[66]:


x = ['KNN','Naive Bayes']
y = [F1_Score_of_KNN, F1_Score_of_Naive_Bayes]
y


# In[67]:


plt.figure(figsize = (8,8))
plt.bar(x,y, color = ['grey', 'blue'])
plt.xlabel('Models')
plt.ylabel("F1 Score")
plt.title("F1 Score of KNN Model and Naive Bayes")
plt.show()


# In[ ]:




