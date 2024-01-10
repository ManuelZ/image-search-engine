# Image search engine

User-interface and backend for finding images that look similar to an input query image.

### Instructions

Configure

To create a database of descriptors of the images in the input folder:
```
python indexer.py
```

To start the local UI:
```
cd frontend
npm start
```

To start the backend:
```
cd backend
python engine.py
```

### Methods
Bag of visual words with BRISK descriptor and HSV color descriptor (the HSV color descriptor from [this Adrian's post](https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/)).