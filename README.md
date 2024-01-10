# Image search engine

User-interface and backend for finding images that look similar to an input query image.

### Methods
- Bag of Visual Words with Daisy corner descriptors, using the Faiss KMeans clusterer.
- Faiss-based Flat Index for querying.

Bag of Visual Words is no longer an effective method for finding similar images; it's primarily a learning exercise.

Additional descriptors, such as the HSV color descriptor mentioned in [PyImagesearch](https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv), are prepared for integration but haven't been incorporated yet.


### Instructions

Configure the path towards the local images in `config.py`.

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

