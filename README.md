# GitHub Avatars

A machine learning experiment to differentiate default GitHub avatars from custom ones.

| example default avatar | example custom avatar | 
| ------- | ------ |
| ![default](avatars/default/abrim.png) | ![custom](avatars/custom/zeke.png) |

## Approach

Create a color histogram for each image by reducing it to 8 bit (or 4 bit? 3 bit?). Traditional learning models suffer from the "curse of dimensionality", wherein the higher the dimensionality the harder to learn. (Not so for deep learning). 768 values (256 * 3) is actually quite a lot of dimensions for a small dataset. There is a connection between the size of dataset and inputs. Small dataset: use fewer inputs.

1. create a data pipeline that reads in images
1. input is image and a label (default or not)
1. output is a 24-valued histogram plus the label

- Start with scikit-learn. Then maybe use torch.
- scikit-learn is easy to start. If it works, great! Otherwise we can switch to torch and use scikit-learn implementation as a baseline.
- It's easy to make bugs in torch
- It's harder to make bugs in scikit

## Running it

```
script/start
```

Jupyter will output a URL to visit in your browser.