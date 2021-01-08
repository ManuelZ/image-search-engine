from pathlib import Path
import logging
import cv2
from descriptors import *


# Logging level
LOGGING_LEVEL = logging.DEBUG

# Logging format
LOGGING_FORMAT = '%(levelname)-5s: @%(funcName)-25s | %(message)s'

# Size of the thumbnails returnd by the Flask server
THUMBNAIL_SIZE = 256

# Before feature extraction images will be resized to this wxh
RESIZE_WIDTH = 250

# Path towards the data directory
# DATA_FOLDER_PATH = Path("C:/Users/Manuel/Desktop/Documentos/Upwork 2021/German team\mastermAInd-extraction")
DATA_FOLDER_PATH = Path("data")

# Which extensions to look for in the data folder
EXTENSIONS = ( '*.jpg', '*.jpeg', '*.png' )

# Path towards the file where all the calculated data is saved
SAVED_DATA_PATH = Path("bovw.joblib") 

# Path towards saved descriptions of images
SAVED_DESCRIPTIONS_PATH = Path("descriptions.joblib")

# Batch size of MiniBatchKmeans (this value is only used when the dataset is 
# larger than the given number)
BATCH_SIZE = 20000

# Number of clusters to test from the valid range of clusters defined
NUM_CLUSTERS_TO_TEST = 1 # if 1, MIN_NUM_CLUSTERS_TO_TEST will be used

# Look for the best number of clusters between these ranges
MIN_NUM_CLUSTERS = 12 # 12 or 29 # minimum is 2
MAX_NUM_CLUSTERS = 100

assert MIN_NUM_CLUSTERS <= MAX_NUM_CLUSTERS, "min n clusters <= max n clusters"
assert BATCH_SIZE >= MAX_NUM_CLUSTERS, "n_samples should be larger than max n_clusters"

# Method to asses the quality of the clustering
CLUSTER_EVAL_METHOD = 'davies-bouldin' # 'silhouette', 'davies-bouldin', 'calinski_harabasz_score'

# Whether to return all the available scores methods
CLUSTER_RETURN_ALL = True

# Size of the sample to evaluate the clustering method
CLUSTER_EVAL_SAMPLE_SIZE = 10000

# Number of times to repeat the sampling
CLUSTER_EVAL_N_SAMPLES = 10     

DESCRIPTORS = {
    "corners" : CornerDescriptor("daisy"),
    #"hog"     : HOGDescriptor()
}