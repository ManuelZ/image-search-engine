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
import skimage
from skimage.util import img_as_ubyte

# Local imports
from config import Config
from descriptors import DESCRIPTORS
from utils import get_image

app = Flask(__name__)
CORS(app)

config = Config()

if config.BOVW_PATH.exists():
    if "corners" in DESCRIPTORS:
        saved = joblib.load(str(config.BOVW_PATH))

        clusterer = saved["clusterer"]
        paths_to_images = saved["images_paths"]
        db_images_features = saved["images_features"]
        pipeline = saved["pipeline"]

        print(len(paths_to_images))

        n_clusters = clusterer.cluster_centers_.shape[0]
    
    else:
        db_images_features, paths_to_images = joblib.load(str(config.BOVW_PATH))


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
        npimg = np.frombuffer(image_str, np.uint8)

        # Convert numpy array to image
        image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = skimage.transform.resize(
            image          = image, 
            output_shape   = (config.RESIZE_WIDTH,config.RESIZE_WIDTH),
            anti_aliasing  = True,
            mode           = 'constant',
            preserve_range = False
        )
        image = img_as_ubyte(image)
        
        to_concatenate = []
        for key,des in DESCRIPTORS.items():
            
            if key == 'corners':
                description = des.describe(image)

                # Quantize the image descriptor:
                # Predict the closest cluster that each sample belongs to. Each value 
                # returned by predict represents the index of the closest cluster 
                # center in the code book.
                clusters_idxs = clusterer.predict(description)

                # Histogram of image descriptor values
                query_im_histogram, _ = np.histogram(clusters_idxs, bins=n_clusters)
                query_im_histogram    = query_im_histogram.reshape(1, -1)
                bovw_histogram        = pipeline.transform(query_im_histogram).todense()

                to_concatenate.append(bovw_histogram)
            
            else:
                description = des.describe(image).reshape(1,-1)
                to_concatenate.append(description)

        
        query_im_features_conc = np.concatenate(to_concatenate, axis=1)

        # TODO: use a vectorized operation instead
        # scipy.spatial.distance.cdist(XA, XB, 'cosine')
        results = {}
        print(db_images_features.shape)
        for i, image_features in enumerate(db_images_features):
            fetA = np.asarray(image_features).reshape(-1)
            fetB = np.asarray(query_im_features_conc).reshape(-1)
            d = cosine(fetA, fetB)
            results[str(paths_to_images[i])] = d
        
        results = sorted([(v, k) for (k, v) in results.items()])
        results = results[:n_images]
        r = []
        for (dist, path_to_im) in results:
            im = get_image(path_to_im)
            if im is not None:
                r.append((dist, im, path_to_im))
        end = time.time()

        print(f"Took {end - start:.1f} seconds.")

        return Response(
            response = json.dumps({'prediction': r}),
            status   = 200,
            mimetype = "application/json"
        )
        
    return "nothing"


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True) 