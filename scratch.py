import sklearn

from PIL import Image

from sklearn.datasets import load_files

# files = load_files('./avatars')

# for datum in files["data"]:
#     print(datum)


from os import listdir
from os.path import isfile, join
import re
import matplotlib.pyplot as plt

mypath = './avatars/custom/'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

x = []
y = []

for file in files:
    print(file)
    # label = file # file.split('_')[0] # assuming your img is named like this "eight_1.png" you want to get the label "eight"
    # y.append(label)
    # img = plt.imread(file)
    # x.append(img)

# img = PIL.Image.open("image_location/image_name") # This returns an image object   
# img = np.asarray(img) # convert it to ndarray

# from PIL import Image 
# import numpy as np

# img = PIL.Image.open("image_location/image_name") # This returns an image object   
# img = np.asarray(img) # convert it to ndarray