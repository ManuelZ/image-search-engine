# Standard Library imports
from pathlib import Path
import logging


class Config:
    NUM_IMAGES_TO_RETURN = 10

    N_JOBS = 4

    DATA_FOLDER_PATH = FILL THIS PATH

    MODELS_BASE_PATH = Path("models")
    BOVW_CORNER_DESCRIPTIONS_PATH = MODELS_BASE_PATH / "bovw_corner_descriptions.joblib"
    BOVW_KMEANS_INDEX_PATH = MODELS_BASE_PATH / "bovw_kmeans_index.faiss"
    BOVW_PIPELINE_PATH = MODELS_BASE_PATH / "bovw_pipeline.joblib"
    BOVW_INDEX_PATH = MODELS_BASE_PATH / "bovw_index.faiss"

    # Batch size of MiniBatchKmeans (this value is only used when the dataset is
    # larger than the given number)
    BATCH_SIZE = 20000

    # Logging level
    LOGGING_LEVEL = logging.INFO

    # Logging format
    LOGGING_FORMAT = "%(levelname)-5s: @%(funcName)-25s | %(message)s"

    # Size of the thumbnails returnd by the Flask server
    THUMBNAIL_SIZE = 256

    # Before feature extraction images will be resized to this wxh
    RESIZE_WIDTH = 256

    # Which extensions to look for in the data folder
    EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")

    # Number of clusters to test from the valid range of clusters defined
    NUM_CLUSTERS_TO_TEST = 3  # if 1, MIN_NUM_CLUSTERS_TO_TEST will be used
    # Look for the best number of clusters between these ranges
    MIN_NUM_CLUSTERS = 20  # minimum is 2
    MAX_NUM_CLUSTERS = 200

    # assert MIN_NUM_CLUSTERS <= MAX_NUM_CLUSTERS, "min n clusters <= max n clusters"
    # assert BATCH_SIZE >= MAX_NUM_CLUSTERS, "n_samples should be larger than max n_clusters"

    # Method to asses the quality of the clustering
    CLUSTER_EVAL_METHOD = (
        "davies-bouldin"  # 'silhouette', 'davies-bouldin', 'calinski_harabasz_score'
    )

    # Size of the sample to evaluate the clustering method
    CLUSTER_EVAL_SAMPLE_SIZE = 2000

    # Number of times to repeat the sampling
    CLUSTER_EVAL_N_SAMPLES = 10

    CORNER_DESCRIPTOR = "orb"
