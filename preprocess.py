import numpy as np
from PIL import Image

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