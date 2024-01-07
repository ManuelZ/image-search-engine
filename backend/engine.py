# Built-in imports
import json
import time

# External imports
from flask import Flask
from flask import request
from flask import Response
from flask_cors import CORS
import joblib
import numpy as np
import cv2
from scipy.spatial.distance import cosine
from imutils import resize
from skimage.util import img_as_ubyte

# Local imports
from config import Config
from descriptors import DESCRIPTORS
from utils import get_image, hamming

app = Flask(__name__)
CORS(app)
config = Config()






def formdata_file_to_image(file):
    """
    """
    image_str = file.read()
    npimg = np.frombuffer(image_str, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    return image


def quantize(description):
    """
    Quantize the image descriptor.
    Predict the closest cluster that each sample belongs to. Each value 
    returned by predict represents the index of the closest cluster 
    center in the code book.
    """
    clusters_idxs = clusterer.predict(description)

    # Histogram of image descriptor values
    query_im_histogram, _ = np.histogram(clusters_idxs, bins=n_clusters)
    query_im_histogram = query_im_histogram.reshape(1, -1)
    bovw_histogram = pipeline.transform(query_im_histogram).todense()

    return bovw_histogram


def run_image_query(query_im_features_conc, n_images):
    """
    """

    # TODO: use a vectorized operation instead
    # scipy.spatial.distance.cdist(XA, XB, 'cosine')
    results = []
    for i, image_features in enumerate(db_images_features):
        fetA = np.asarray(image_features).reshape(-1)
        fetB = np.asarray(query_im_features_conc).reshape(-1)

        assert fetA.shape[0] == fetB.shape[0], "Descriptions have different number of rows"
        assert fetA.shape[1] == fetB.shape[1], "Descriptions have different number of columns"
        
        # If using dhash
        if fetA.shape[0] == 1:
            d = hamming(fetA, fetB)
        else:
            d = cosine(fetA, fetB)
        
        results.append((str(paths_to_images[i]), d))
    
    # Arrange in ascending order
    results.sort(key=lambda x: x[1])
    
    # Keep only the top n results
    results = results[:n_images]
    
    predictions = []
    for (path_to_im, dist) in results:
        im = get_image(path_to_im)
        if im is not None:
            predictions.append((dist, im, path_to_im))

    return predictions


def extract_features(image):
    """
    """
    
    to_concatenate = []
    for (descriptor_name, descriptor) in DESCRIPTORS.items():
        description = descriptor.describe(image)
        if descriptor_name == 'corners':
            description = quantize(description)  # bovw histogram
        else:
            description = description.reshape(1,-1)
        to_concatenate.append(description)

    query_im_features_conc = np.concatenate(to_concatenate, axis=1)

    return query_im_features_conc


@app.route('/similar_images', methods=['POST'])
def predict():
    """
    """

    if not request.files:
        return Response("No file uploaded", status=400)

    start = time.time()
    image = formdata_file_to_image(request.files['image'])
    image = resize(image, width=config.RESIZE_WIDTH)
    image = img_as_ubyte(image)
    query_im_features_conc = extract_features(image)
    predictions = run_image_query(query_im_features_conc, config.NUM_IMAGES_TO_RETURN)
    end = time.time()

    print(f"Took {end - start:.1f} seconds.")

    return Response(
        response = json.dumps({'prediction': predictions}),
        status   = 200,
        mimetype = "application/json"
    )


if __name__ == "__main__":

    if not config.BOVW_PATH.exists():
        raise Exception("BOVW_PATH not found")

    saved = joblib.load(str(config.BOVW_PATH))
    paths_to_images = saved["images_paths"]
    db_images_features = saved["images_features"]

    if "corners" in DESCRIPTORS:
        clusterer = saved["clusterer"]
        paths_to_images = saved["images_paths"]
        db_images_features = saved["images_features"]
        pipeline = saved["pipeline"]
        n_clusters = clusterer.cluster_centers_.shape[0]
    
    app.run(host='127.0.0.1', port=5000, debug=True)
