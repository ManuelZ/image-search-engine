# Standard Library imports
from pathlib import Path
import logging


class Config:
    NUM_IMAGES_TO_RETURN = 10

    MULTIPROCESS = True

    N_JOBS = 6

    DATA_FOLDER_PATH = Path(r"F:\DATASETS\102 Flowers dataset\102flowers\jpg")

    # Path towards saved descriptions of images
    DESCRIPTIONS_PATH = Path("descriptions.joblib")

    BOVW_CODEBOOK_PATH = Path("bovw_codebook.joblib")
    BOVW_PIPELINE_PATH = Path("bovw_pipeline.joblib")
    BOVW_CORNER_DESCRIPTIONS_PATH = Path("bovw_corner_descriptions.joblib")
    BOVW_HISTOGRAMS_PATH = Path("bovw_histograms.npy")
    BOVW_INDEX_PATH = Path("index.index")

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
    NUM_CLUSTERS_TO_TEST = 1  # if 1, MIN_NUM_CLUSTERS_TO_TEST will be used
    # Look for the best number of clusters between these ranges
    MIN_NUM_CLUSTERS = 10  # 12 or 29 # minimum is 2
    MAX_NUM_CLUSTERS = 1000

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

    CORNER_DESCRIPTOR = "daisy"