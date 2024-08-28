# Standard Library imports
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum


class Method(Enum):
    BOVW = 1
    DNN = 2
    DHASH = 3


class DnnModels(Enum):
    RESNET = 1
    BiT = 2


@dataclass
class Config:

    ####################################################################################################################
    # General config
    ####################################################################################################################

    # Logging level
    LOGGING_LEVEL = logging.INFO

    # Logging format
    LOGGING_FORMAT = "%(levelname)-5s: @%(funcName)-25s | %(message)s"

    # Before feature extraction, images will be resized to this size
    RESIZE_SIZE = 224  # height or width, use as it most convenient

    # Which extensions to look for in the data folder
    EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

    # Split config into inference and indexer
    NUM_IMAGES_TO_RETURN = 20

    # For n_jobs below -1, (n_cpus + 1 + n_jobs) are used
    # 1: no parallel computing code is used at all
    # -1: all cpus
    N_JOBS = 1

    DATA_FOLDER_PATH = FILL THIS PATH

    # Path towards the folder where models are saved
    MODELS_BASE_PATH = Path("models")

    # Size of the thumbnails returnd by the Flask server
    THUMBNAIL_SIZE = 256

    DEVICE = "cuda"

    ####################################################################################################################
    # Features and index
    ####################################################################################################################

    # What method to use for feature extraction
    METHOD = Method.BOVW

    INDEX_TYPE = "l2"  # cosine, l2, cell-probe

    ####################################################################################################################
    # Dhash config
    ####################################################################################################################
    DHASH_INDEX_PATH = MODELS_BASE_PATH / "dhash_index.pickle"

    ####################################################################################################################
    # DNN config
    ####################################################################################################################

    DNN_MODEL = DnnModels.RESNET

    DNN_INDEX_PATH = MODELS_BASE_PATH / "resnet50_dnn_index.faiss"  # "dnn_index.faiss"

    ####################################################################################################################
    # BOVW config
    ####################################################################################################################

    # Flag to turn on/off the BOVW hyperparameters search
    BOVW_HYPERPARAMETERS_SEARCH = False

    CORNER_DESCRIPTOR = "orb"  # brisk, sift, orb
    BOVW_CORNER_DESCRIPTIONS_PATH = MODELS_BASE_PATH / "bovw_corner_descriptions.joblib"
    BOVW_KMEANS_INDEX_PATH = MODELS_BASE_PATH / "bovw_kmeans_index.faiss"
    BOVW_PIPELINE_PATH = MODELS_BASE_PATH / "bovw_pipeline.joblib"
    BOVW_INDEX_PATH = MODELS_BASE_PATH / "bovw_index.faiss"

    # Method to asses the quality of the clustering
    CLUSTER_EVAL_METHOD = (
        "davies-bouldin"  # 'silhouette', 'davies-bouldin', 'calinski_harabasz_score'
    )

    # Size of the sample to evaluate the clustering method
    CLUSTER_EVAL_SAMPLE_SIZE = 2000

    # Number of times to repeat the sampling
    CLUSTER_EVAL_N_SAMPLES = 10

    # Number of clusters to use if no hyperparameter search is performed
    NUM_CLUSTERS = 200

    # Number of clusters to test from the valid range of clusters defined
    NUM_CLUSTERS_TO_TEST = 3  # if 1, MIN_NUM_CLUSTERS_TO_TEST will be used
    # Look for the best number of clusters between these ranges
    MIN_NUM_CLUSTERS = 20  # minimum is 2
    MAX_NUM_CLUSTERS = 200
