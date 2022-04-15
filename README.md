# GitHub Avatars

A machine learning model to differentiate default GitHub avatars from custom ones.

| example default avatar | example custom avatar | 
| ------- | ------ |
| ![default](avatars/default/abrim.png) | ![custom](avatars/custom/zeke.png) |


## Development

First, install [Cog](https://github.com/replicate/cog#install). Then:

```
script/start
```

Jupyter will output a URL to visit in your browser.

## Build

Saving any changes you've made to the notebook.

If you've never run the build before:

```
cog build
```

Then:

```
script/build
```

This will convert the notebook to a python script and build the cog image.

## Usage

### Training

Run the notebook, and run all the cells. This will output an updated `model.pkl`.

### Predicting

Call `cog predict` and specify a GitHub username as input to get a prediction:

```
cog predict -i username=zeke
```

You should see output like this:

```json
{
  "username": "zeke",
  "href": "https://github.com/zeke.png",
  "prediction": "custom"
}
```

---

## Notes

- Start with scikit-learn. Then maybe use torch.
- scikit-learn is easy to start. If it works, great! Otherwise we can switch to torch and use scikit-learn implementation as a baseline.
- It's easy to make bugs in torch
- It's harder to make bugs in scikit

### Approach (v1)

Create a color histogram for each image by reducing it to 8 bit (or 4 bit, or 3). Traditional learning models suffer from the "curse of dimensionality", wherein the higher the dimensionality, the harder to learn. (Not so for deep learning). 768 values (256 * 3) is actually quite a lot of dimensions for a small dataset. There is a connection between the size of dataset and inputs. For small datasets, one should use fewer inputs.

1. create a data pipeline that reads in images
1. input is image and a label (default or not)
1. output is a 24-valued histogram plus the label


### Notes, January 2022

Feature Engineering is a process of manually constructing features that suit that task at hand. Our current feature is a color histogram, counting the number of colors in each image. But we could also contruct features in a different way, for example coundting the number of unique colors in each image.

Feature engineering is common in traditional ML, but in deep learning the emphasis is more on the learning of the model itself, rather than the learning of the features. Specifically, finding the right model architecture. Feature engineering isn't really used any more.

Today we'll focus on understanding why the model makes the deicsions it makes, based on the features we've constructed, and hopefully improve our feature engineering step based on what we learn.

### Approach (v2)

v1 was flawed, in our quantizing process caused the "github grey" color to be bucketed along with "white", leading to some custom avatars being misidentified as default avatars.

In this new version, we'll create a new feature vector using:

- the number of "GitHub grey" pixels in the image
- the number of distinct colors in the image


