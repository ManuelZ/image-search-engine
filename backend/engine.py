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
from bag_of_visual_words import load_cluster_model
from descriptors import Describer, CornerDescriptor

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
    tmp_save_path = ".tmp.png"
    cv2.imwrite(tmp_save_path, image)
    image_features = pipeline.transform(np.array([tmp_save_path])).todense()
    predictions = run_image_query(image_features, config.NUM_IMAGES_TO_RETURN)
    end = time.time()

    print(f"Took {end - start:.2f} seconds.")

    return Response(
        response=json.dumps({"prediction": predictions}),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":
    pipeline = joblib.load(str(config.BOVW_PIPELINE_PATH))

    n_clusters = pipeline.named_steps["bovw"].n_clusters
    print(f"n_clusters: {n_clusters}")

    # Load images paths
    images_paths = get_images_paths()

    # Load KMeans
    clusterer = load_cluster_model(n_clusters, config.BOVW_KMEANS_INDEX_PATH)
    pipeline.named_steps["bovw"].clusterer = clusterer

    # Computer and/or load the BOVW corner descriptors
    describer = Describer({"corners": CornerDescriptor(config.CORNER_DESCRIPTOR)})

    # This index is no the clusterer index
    index = faiss.read_index(str(config.BOVW_INDEX_PATH))
    print(f"There are {index.ntotal} images in the index.")

    app.run(host="127.0.0.1", port=5000, debug=True)
