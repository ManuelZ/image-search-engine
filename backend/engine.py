# Built-in imports
import json
import time
import pickle

# External imports
from flask import Flask
from flask import request
from flask import Response
from flask_cors import CORS
import numpy as np
import cv2
import faiss
import joblib

# Local imports
from config import Config, Method, DnnModels
from descriptors import CornerDescriptor
from utils import get_image, get_images_paths
from bag_of_visual_words import load_cluster_model
from descriptors import Describer, CornerDescriptor, CNNDescriptor, DHashDescriptor
import torch

config = Config()
app = Flask(__name__)
CORS(app)

if config.METHOD == config.METHOD.DNN:
    descriptor = CNNDescriptor(model=config.DNN_MODEL)
elif config.METHOD == config.METHOD.BOVW:
    descriptor = CornerDescriptor(config.CORNER_DESCRIPTOR)
elif config.METHOD == config.METHOD.DHASH:
    descriptor = DHashDescriptor()
else:
    raise Exception("Method can only be DNN or BOVW")


def formdata_file_to_image(file):
    """ """
    image_str = file.read()
    npimg = np.frombuffer(image_str, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return image


def run_image_query(image_features, n_images, normalize=False):
    """ """

    if isinstance(image_features, torch.Tensor):
        image_features = image_features.detach().cpu().numpy().reshape(1, -1)

    if normalize:
        faiss.normalize_L2(image_features)

    distances, indices = index.search(image_features, n_images)
    distances = distances.ravel().tolist()
    indices = indices.ravel().tolist()

    predictions = []
    for dist, i in zip(distances, indices):
        image_path = images_paths[i]
        image = get_image(image_path)
        predictions.append((dist, image, str(image_path)))

    return predictions


@app.route("/similar_images", methods=["POST"])
def predict():
    """ """

    if not request.files:
        return Response("No file uploaded", status=400)

    image = formdata_file_to_image(request.files["image"])

    start = time.time()
    if config.METHOD == Method.DNN:
        image_features = descriptor.describe(image)
        predictions = run_image_query(image_features, config.NUM_IMAGES_TO_RETURN)

    elif config.METHOD == Method.DHASH:
        dhash = image_features[0][0]
        print(f"Query: {dhash}")
        image_paths = index.get(dhash, [])
        print(f"Found: ", image_paths)

        predictions = []
        for image_path in image_paths:
            image = get_image(image_path)
            predictions.append((0, image, str(image_path)))

    elif config.METHOD == Method.BOVW:
        tmp_save_path = ".received.png"
        cv2.imwrite(tmp_save_path, image)
        bovw_histograms = pipeline.transform(np.array([tmp_save_path])).todense()
        bovw_histograms = bovw_histograms.astype(np.float32)
        predictions = run_image_query(bovw_histograms, config.NUM_IMAGES_TO_RETURN)

    end = time.time()
    print(f"Took {end - start:.2f} seconds.")

    return Response(
        response=json.dumps({"prediction": predictions}),
        status=200,
        mimetype="application/json",
    )


if __name__ == "__main__":

    images_paths = get_images_paths()

    if config.METHOD == config.METHOD.DNN:
        print(f"Loading index from '{str(config.DNN_INDEX_PATH)}'")
        index = faiss.read_index(str(config.DNN_INDEX_PATH))
        print(f"There are {index.ntotal} images in the index.")

    elif config.METHOD == config.METHOD.BOVW:

        # Load the pipeline
        pipeline = joblib.load(str(config.BOVW_PIPELINE_PATH))

        # Load the clusterer into the pipeline
        n_clusters = pipeline.named_steps["bovw"].n_clusters
        clusterer = load_cluster_model(n_clusters, config.BOVW_KMEANS_INDEX_PATH)
        pipeline.named_steps["bovw"].clusterer = clusterer
        print(f"n_clusters: {n_clusters}")

        # Load the index
        index = faiss.read_index(str(config.BOVW_INDEX_PATH))

    elif config.METHOD == config.METHOD.DHASH:
        with open(config.DHASH_INDEX_PATH, "rb") as f:
            index = pickle.load(f)

    app.run(host="127.0.0.1", port=5000, debug=True)
