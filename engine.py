# Built-in imports
import sys
import json
import time
import io
import base64

# External imports
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
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

# Local imports
from config import SAVED_DATA_PATH
from bag_of_visual_words  import compute_image_descriptors
from bag_of_visual_words import chi2_distance


app = Flask(__name__)
CORS(app)

def get_image(image_path):
    size = 128, 128
    img = Image.open(image_path, mode='r')
    img.thumbnail(size, Image.ANTIALIAS)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return encoded_img

if SAVED_DATA_PATH.is_file():
    (kmeans, k, tfidf, scaler, neighbors, codebook, tfidf_hist_scaled_features, paths_to_images) = joblib.load(str(SAVED_DATA_PATH))

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

        # Compute image descriptors
        kp, des = compute_image_descriptors(image)

        # Quantize the image descriptor
        quantized_desc = codebook[kmeans.predict(des)]

        # Histogram of image descriptor values
        im_feat_hist, _ = np.histogram(quantized_desc, bins=k)
        im_feat_hist = im_feat_hist.reshape(1, -1)

        # TF-IDF
        im_feat_hist_tfidf = tfidf.transform(im_feat_hist)
                
        # Scale the histogram
        im_scaled_feat_hist_tfidf = scaler.transform(im_feat_hist_tfidf)
        
        # Find the neighbors of the input
        # distances, indices = neighbors.kneighbors(im_scaled_feat_hist, n_neighbors=n_images)
        # results = np.array(paths_to_images)[indices[0]]
        # results = [str(x) for x in results]

        results = {}
        for i, features in enumerate(tfidf_hist_scaled_features):
            # Sparse -> Matrix -> 1D array
            histA = np.asarray(features.todense()).reshape(-1)
            histB = np.asarray(im_scaled_feat_hist_tfidf.todense()).reshape(-1)

            d = chi2_distance(histA, histB)
            results[str(paths_to_images[i])] = d
        
        results = sorted([(v, k) for (k, v) in results.items()])
        results = results[:n_images]
        results = [(r[0], get_image(r[1])) for r in results]

        end = time.time()

        print(f"Took {end - start:.1f} seconds.")

        return Response(response=json.dumps({'prediction': results}), status=200, mimetype="application/json")
        
    return "nothing"


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True) 