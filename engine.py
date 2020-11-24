# Built-in imports
import sys
import json
import time
import io
import base64

# External imports
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from flask import Flask
from flask import request
from flask import Response
from flask import send_file
from flask_cors import CORS, cross_origin
import pandas as pd
import joblib
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.distance import cosine

# Local imports
from config import SAVED_DATA_PATH
from config import THUMBNAIL_SIZE
from config import RESIZE_WIDTH
from bag_of_visual_words import resize
from descriptors import ColorDescriptor
from descriptors import CornerDescriptor

app = Flask(__name__)
CORS(app)

def get_image(image_path):
    size = THUMBNAIL_SIZE, THUMBNAIL_SIZE
    img = Image.open(image_path, mode='r')
    img.thumbnail(size, Image.ANTIALIAS)
    img_byte_arr = io.BytesIO()
    try:
        img.save(img_byte_arr, format='JPEG')
    except OSError:
        img.save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img

if SAVED_DATA_PATH.exists():
    (
        kmeans, # Trained Kmeans that is able to predict the closest cluster each sample in X belongs to.
        n_clusters,
        codebook,
        scaler,
        tfidf,
        images_features,
        paths_to_images
    ) = joblib.load(str(SAVED_DATA_PATH))

pipeline = Pipeline([
    ('scaler', scaler),
    ('tfidf', tfidf),
])

corner_des = CornerDescriptor()
color_des  = ColorDescriptor()

@app.route('/similar_images', methods=['POST'])
def predict():
    """
    """
    start = time.time()

    # Number of similar images to return
    n_images = 10

    if request.files:
        # Retrieve the posted image as a string
        image_str = request.files['image'].read()

        # Convert string data to numpy array
        npimg = np.fromstring(image_str, np.uint8)

        # Convert numpy array to image
        image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        
        # Resize to standard custom size
        image = resize(image, RESIZE_WIDTH)

        # Compute image descriptors
        im_corner_descriptors = corner_des.describe(image)
        im_color_descriptors  = color_des.describe(image)

        # Quantize the image descriptor:
        # Predict the closest cluster that each sample belongs to. Each value 
        # returned by predict represents the index of the closest cluster 
        # center in the code book.
        clusters_idxs = kmeans.predict(im_corner_descriptors)

        # Histogram of image descriptor values
        query_im_histogram, _ = np.histogram(clusters_idxs, bins=n_clusters)
        query_im_histogram    = query_im_histogram.reshape(1, -1)
        query_im_histogram    = pipeline.transform(query_im_histogram)

        query_im_color_features = np.array(im_color_descriptors).reshape((1,-1))
        query_im_features_conc = np.concatenate([
            query_im_histogram.todense(), query_im_color_features
        ], axis=1)

        results = {}
        for i, image_features in enumerate(images_features):
            histA = np.asarray(image_features).reshape(-1)
            histB = np.asarray(query_im_features_conc).reshape(-1)
            d = cosine(histA, histB)
            results[str(paths_to_images[i])] = d
        
        results = sorted([(v, k) for (k, v) in results.items()])
        results = results[:n_images]
        results = [(dist, get_image(path_to_im), path_to_im) for (dist, path_to_im) in results]

        end = time.time()

        print(f"Took {end - start:.1f} seconds.")

        return Response(
            response=json.dumps({'prediction': results}),
            status=200,
            mimetype="application/json"
        )
        
    return "nothing"


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True) 