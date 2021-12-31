#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn import tree
from sklearn.datasets import load_files
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# In[2]:


files = load_files('./avatars', load_content=False)


# In[9]:


files.keys()


# In[10]:


# labels
files["target"]


# In[12]:


# folder names
files["target_names"]


# In[3]:


def file_to_color_histogram(path):
    
    # open the file with pillow, resize it, normalize
    image = Image.open(path)
    image = image.resize((16, 16))
    image = image.convert("RGB")

    # use numpy to extra an array of pixel data from the image
    data = image.getdata()
    pix = np.array(data)

    # divide each pixel by 32 and round
    pix = pix//32 

    # convert octal numbers to decimal numbers
    multiplier = np.array([8**2, 8**1, 8**0])

    transformed_pixel_values = np.sum(pix*multiplier, axis=1)

    feature_vector_omg = np.bincount(transformed_pixel_values, minlength=512)

    return feature_vector_omg


# In[5]:


# each histogram is an array of 8bits^3 (512 values) corresponding to the 8-bit colors in each image 
# and each element in the array is the count of the number of pixels of that color
histograms = []
for file in files["filenames"]:
    histogram = file_to_color_histogram(file)
    histograms.append(histogram)


# In[7]:


histograms[0]


# In[6]:


histograms[0].shape


# In[16]:


# split our histograms and labels into a training™ set and a validation™ set
# use part of our dataset as a "holdout set" during training to get a sense 
# of how it will work on ral images

# sometimes there's a "test" set that's used when you're developing the model to avoid overoptimizing for the test set, but we're not using it.

# we'll use half as training, half as validation

# TODO: considering balancing these sets so they have an equal number of 
# default and custom avatars
training_set = histograms[0:len(histograms)//2]
validation_set = histograms[len(histograms)//2:]

labels = files["target"]
training_labels = labels[0:len(labels)//2]
validation_labels = labels[len(labels)//2:]

[len(training_set), len(validation_set), len(training_labels), len(validation_labels)]


# In[21]:


# start with a simple decision tree
# looks at all the data, finds the split that makes one group have most of one label, and another
# group have most of another label

# finds one feature, like the color black. If the number of black pixels is more than 100, that means 80% of 
# the custom images end up in one bucket. 

# (if that is the most discriminating feature that can split that feature into two, it picks that feature))

# find the feature that splits it most coherently into two groups
# then find the second-most splitting feature

# the "feature" in this case will the number of pixels of a specific color
# in this case, likely the gray background of the default avatars

clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_set, training_labels)
tree.plot_tree(clf)


# In[ ]:


tree.plot_tree(clf)


# In[40]:


clf.tree_


# In[23]:


# gini coefficient: goodness measure of how coherent the two groups are (zero is perfect, one (or 0.5) is useless)
# it's what the decision tree uses to determine how to split

# root node: if index 511 fewer or equal to than 67.5 of that color, split into two groups
# group on the left: 33 custom images and no default images
# group on the right: 7 custom, 60 default

# TODO: write a little function that turns an index into a hex color so we can see what it looks like


# throw all of the validation examples through the tree and see which nodes they end up with
prediction = clf.predict(validation_set[:1])[0]
prediction


# In[29]:


validation_set[0][511], validation_set[0][438], validation_set[0][475]


# In[30]:


predictions = clf.predict(validation_set)
predictions


# In[32]:


np.equal(predictions, validation_labels)


# In[33]:


np.mean(np.equal(predictions, validation_labels))


# In[36]:


# download image, convert to histogram, call clf.predict on [histogram]
histogram = file_to_color_histogram("smoke-tests/andreasjansson.png")
prediction = clf.predict([histogram])[0]
prediction


# In[37]:


histogram = file_to_color_histogram("smoke-tests/default.png")
prediction = clf.predict([histogram])[0]
prediction


# In[ ]:




