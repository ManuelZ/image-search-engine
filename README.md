# Image search engine

User-interface and backend for finding images that look similar to an input query image.

### Instructions

Run the following to create a database of descriptors of the images in the input folder:
    python bag_of_visual_words.py

In one console:

    cd image-search-engine-ui
    npm start

In another console:

    python engine


### Methods
Bag of visual words with BRISK descriptor and HSV color descriptor (the HSV color descriptor from [this Adrian's post](https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/)).