#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import tree
from sklearn.datasets import load_files
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# In[2]:


# deterministic, it seems
files = load_files('./avatars', load_content=False)


# In[3]:


files.keys()


# In[4]:


# labels
files["filenames"][0:5], files["target_names"], files["target"][0:10]


# In[5]:


# folder names
files["target_names"]


# In[6]:


def get_rgb_array(path):
    """
    open the image with pillow and use numpy to extra an array of RGB pixel data
    """
    image = Image.open(path)
    image = image.resize((16, 16), resample=Image.NEAREST)
    image = image.convert("RGB")
    data = image.getdata()
    return np.array(data)


def get_unique_color_count(rgb_array):
    """get unique rows (axis=0 is row, axis=1 is column)"""
    unique_colors = np.unique(rgb_array, axis=0)
    return len(unique_colors)

def get_github_grey_count(rgb_array):
    """count the number of pixels that are GitHub Grey™"""
    github_grey = [240, 240, 240]

    # we have a two-dimensional array where each row is a RGB array
    # np.all() returns an array of booleans where each value is true if 
    # all of the values in the corresponding row match GitHub grey.
    #
    # PS numpy overrides the equality operator! 🤯
    return np.sum(np.all(rgb_array == github_grey, axis=1))

def file_to_feature_vector(path):
    rgb_array = get_rgb_array(path)
    unique_color_count = get_unique_color_count(rgb_array)
    grey_count = get_github_grey_count(rgb_array)
    return [grey_count, unique_color_count]


# In[7]:


some_default_avatar = files["filenames"][1]
rgb_array = get_rgb_array(some_default_avatar)


get_github_grey_count(rgb_array)


# In[8]:


examples = [file_to_feature_vector(path) for path in files["filenames"]]

examples


# In[9]:


# split our examples and labels into a "training" set and a "validation" set
# use part of our dataset as a "holdout set" during training to get a sense of how it will work on real images

# (sometimes there's a "test" set that's used when you're developing the model to avoid overoptimizing 
# for the validation set, but we're not using it.)

# we'll use half as training, half as validation

# TODO: considering balancing these sets so they have an equal number of default and custom avatars

training_set = examples[0:len(examples)//2]
validation_set = examples[len(examples)//2:]

labels = files["target"]
training_labels = labels[0:len(labels)//2]
validation_labels = labels[len(labels)//2:]

[len(training_set), len(validation_set), len(training_labels), len(validation_labels)]


# In[10]:


# find the feature and the threshold that best splits this into two classes

# go through all the examples. for all of the two features we have.

# Try a threshold until it finds a value that most cleanly splits the data.

# start with a simple decision tree
# looks at all the data, finds the split that makes one group have most of one label, and another
# group have most of another label

# (if that is the most discriminating feature that can split that feature into two, it picks that feature))

# find the feature that splits it most coherently into two groups
# then find the second-most splitting feature

clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_set, training_labels)
tree.plot_tree(clf)


# In[11]:


# gini coefficient: goodness measure of how coherent the two groups are (zero is perfect, one (or 0.5?) is useless)
# it's what the decision tree uses to determine how to split

# throw all of the validation examples through the tree and see which nodes they end up with
prediction = clf.predict(validation_set[:1])[0]
prediction


# In[12]:


predictions = clf.predict(validation_set)
predictions


# In[13]:


# indices of avatars that are actually defaults but are classified as custom

# single &: elementwise boolean AND of vectors of matrices

# 1: default
# 0: custom

false_customs = np.where((predictions != validation_labels)&(validation_labels == 1))[0]
false_customs


# In[14]:


# indices of avatars that are actually customs but are classified as defaults
# 1: default
# 0: custom

false_defaults = np.where((predictions != validation_labels)&(validation_labels == 0))[0]
false_defaults


# In[15]:


# what's our accuracy?
np.mean(np.equal(predictions, validation_labels))


# In[16]:


# try it out on a custom avatar
feature_vector = file_to_feature_vector("smoke-tests/andreasjansson.png")
prediction = clf.predict([feature_vector])[0]
prediction


# In[17]:


# try it out on a default avatar
feature_vector = file_to_feature_vector("smoke-tests/default.png")
prediction = clf.predict([feature_vector])[0]
prediction

