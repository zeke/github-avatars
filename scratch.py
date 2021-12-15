import sklearn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sklearn.datasets import load_files

files = load_files('./avatars', load_content=False)

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

histograms = []
for file in files["filenames"]:
    histogram = file_to_color_histogram(file)
    histograms.append(histogram)

histograms