#!/usr/bin/env python
# coding: utf-8

#  # Scikit-learn

# In[1]:


pip install -U scikit-learn


# In[2]:


pip install scipy


# In[3]:


pip install numpy


# In[4]:


pip install matplotlib


# In[6]:


import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
print(digits.data)


# In[7]:


digits= datasets.load_digits()
print(digits.target)
print(digits.images[0])


# In[13]:


digits= datasets.load_digits()                     
clf = svm.SVC(gamma=0.001, C=100)
print(len(digits.data))
x,y=digits.data[:-1],digits.target[:-1]            
clf.fit(x,y)
print('Prediction:', clf.predict(digits.data[-1:])) 
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()


# In[16]:


digits= datasets.load_digits()
# Join the images and target labels in a list
images_and_labels = list(zip(digits.images, digits.target))

# for every element in the list
for index, (image, label) in enumerate(images_and_labels[:8]):
    # initialize a subplot of 2X4 at the i+1-th position
    plt.subplot(2, 4, index + 1)
    # Display images in all subplots
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    # Add a title to each subplot
    plt.title('Training: ' + str(label))
    
# Show the plot
plt.show()


# In[ ]:




