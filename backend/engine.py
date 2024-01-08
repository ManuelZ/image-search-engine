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
from imutils import resize
from skimage.util import img_as_ubyte
import faiss

# Local imports
from config import Config
from descriptors import CornerDescriptor
from utils import get_image, get_images_paths
from bag_of_visual_words import (
    load_cluster_model,
    extract_features,
)

app = Flask(__name__)
CORS(app)
config = Config()


def formdata_file_to_image(file):
    """ """
    image_str = file.read()
    npimg = np.frombuffer(image_str, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    return image


def run_image_query(image_features, n_images):
    """ """

    L2_distances, indices = index.search(image_features, n_images)

    L2_distances = L2_distances.ravel().tolist()
    indices = indices.ravel().tolist()

    predictions = []
    for dist, i in zip(L2_distances, indices):
        image_path = images_paths[i]
        image = get_image(image_path)
        predictions.append((dist, image, str(image_path)))

    return predictions





@app.route("/similar_images", methods=["POST"])
def predict():
    """ """

    if not request.files:
        return Response("No file uploaded", status=400)

    start = time.time()
    image = formdata_file_to_image(request.files["image"])
    image = resize(image, width=config.RESIZE_WIDTH)
    image = img_as_ubyte(image)
    image_features = extract_features(image, clusterer, pipeline)
    predictions = run_image_query(image_features, config.NUM_IMAGES_TO_RETURN)
    end = time.time()

    print(f"Took {end - start:.1f} seconds.")

    return Response(
        response=json.dumps({"prediction": predictions}),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    clusterer = load_cluster_model(config.BOVW_INDEX_PATH)
    pipeline = joblib.load(str(config.BOVW_PIPELINE_PATH))
    bovw_histograms = np.load(str(config.BOVW_HISTOGRAMS_PATH))

    d = bovw_histograms.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(bovw_histograms)  # add vectors to the index
    print(f"There are {index.ntotal} images in the index.")
    images_paths = get_images_paths()

    app.run(host="127.0.0.1", port=5000, debug=True)
