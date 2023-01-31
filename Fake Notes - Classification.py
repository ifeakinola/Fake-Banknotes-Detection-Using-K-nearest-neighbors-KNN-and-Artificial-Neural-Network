#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


#Load the dataset
dataset = pd.read_csv('data_banknote_authentication.txt', header=None)


# In[3]:


#explore the dataset
dataset.head()


# In[4]:


#add column names to dataframe
dataset.columns = ['variance_of_Wavelet', 'skewness_of_Wavelet', 'curtosis_of_Wavelet', 'entropy_of_image', 'class' ]


# In[5]:


#explore the dataset; number of rows, number of columns, column labels, column data types, 
#the number of non-null values in each column
dataset.info()


# In[6]:


# review the statistical description of the data for numeric columns
dataset.describe()


# In[7]:


# explore the dataframe visually for patterns
plt.figure(figsize=(10,5))
sns.histplot(x='variance_of_Wavelet', hue= 'class', data=dataset)
plt.title('Variance of Wavelet based on class')
plt.show()


# In[8]:


# explore the dataframe visually for patterns
plt.figure(figsize=(10,5))
sns.histplot(x='skewness_of_Wavelet', hue= 'class', data=dataset)
plt.title('skewness of Wavelet based on class')
plt.show()


# In[9]:


# explore the dataframe visually for patterns
plt.figure(figsize=(10,5))
sns.histplot(x='curtosis_of_Wavelet', hue= 'class', data=dataset)
plt.title('Curtosis_of_Wavelet based on class')
plt.show()


# In[10]:


# explore the dataframe visually for patterns
plt.figure(figsize=(10,5))
sns.histplot(x='entropy_of_image', hue= 'class', data=dataset)
plt.title('Entropy of Image')
plt.show()


# In[11]:


#Check for outliers
ax = dataset.boxplot(figsize=(10,5))
ax.set_xlabel('Features')
ax.set_ylabel('Values')
ax.set_title('Outliers detection')
plt.show()


# In[12]:


#Check visually if the dataset is balanced with regards to the class
dataset['class'].value_counts().plot(kind= 'bar');


# In[13]:


#Check if the dataset is balanced with regards to the class
dataset['class'].value_counts()


# In[14]:


#Slice the data into input and output
X = dataset.iloc[:, [0,1,2,3]].values
y = dataset.iloc[:, 4].values


# In[15]:


#Split the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[16]:


# scale the train and test dataset to a standard range
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train_s = sc.fit_transform(X_train)
X_test_s=sc.transform(X_test)


# In[17]:


# Fitting KNN to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5, metric= 'minkowski', p=2)
classifier.fit(X_train_s, y_train)


# In[18]:


# predicting the Test set results
y_pred=classifier.predict(X_test_s)
print(y_pred)


# In[19]:


print(y_test)


# In[20]:


# Evaluate the performance of the model
from sklearn import metrics
acc=metrics.accuracy_score(y_test, y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm=metrics.confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm, '\n\n')
print('---------------------------------------')
result=metrics.classification_report(y_test, y_pred)
print('Classification Report:\n')
print(result)
ax = sns.heatmap(cm, cmap='flare',annot=True, fmt='d')
plt.xlabel("Predicted Class",fontsize=12)
plt.ylabel("True Class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)
plt.show()


# In[21]:


#Using Neural Network
# importing libraries
#pip install tensorflow
import tensorflow as tf
from keras.layers import Dense


# In[22]:


#Divide the dataset into training and test dataset
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.2, random_state= 99)


# In[23]:


#Normalize the dataset
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train1)
X_test1 = scaler.transform(X_test1)


# In[24]:


#Initialize model
model = tf.keras.models.Sequential()


# In[25]:


#Add Layers
model.add(Dense(8,activation='relu',input_shape=(4,)))
model.add(Dense(1,activation='sigmoid'))


# In[26]:


#Compile model
model.compile(optimizer='adam', loss='binary_crossentropy',metrics='accuracy')


# In[27]:


#View the summary of the neural network that has been built
model.summary()


# In[28]:


#Train the model
history = model.fit(X_train1, y_train1, batch_size = 4, epochs = 20, verbose = 2, validation_split=0.2)


# In[29]:


#plot the training and validation accuracy
accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

plt.plot(accuracy, label='Training Set Accuracy')
plt.plot(validation_accuracy, label='Validation Set Accuracy')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy Across Epochs')
plt.legend()


# In[30]:


#plot the training and validation loss across epochs
loss = history.history['loss']
validation_loss = history.history['val_loss']
plt.plot(loss, label='Training Set Loss')
plt.plot(validation_loss, label='Validation Set Loss')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Across Epochs')
plt.legend()


# In[57]:


#Generate predictions for test data
y_pred1 = model.predict(X_test1)
y_pred1 = np.round(y_pred1).astype(int)


# In[58]:


print(y_pred1)


# In[59]:


#Generate confusion matrix to visualize the results
acc1=metrics.accuracy_score(y_test1, y_pred1)
print('accuracy:%.2f\n\n'%(acc1))
cm1=metrics.confusion_matrix(y_test1, y_pred1)
print('Confusion Matrix:')
print(cm1, '\n\n')
print('---------------------------------------')
result1=metrics.classification_report(y_test1, y_pred1)
print('Classification Report:\n')
print(result1)
ax = sns.heatmap(cm1, cmap='flare',annot=True, fmt='d')
plt.xlabel("Predicted Class",fontsize=12)
plt.ylabel("True Class",fontsize=12)
plt.title("Confusion Matrix",fontsize=12)
plt.show()


# In[ ]:





# In[ ]:




