from pathlib import Path
import cv2

# Which extensions to look for
EXTENSIONS = (
  '*.jpg',
  '*.jpeg',
  '*.png',
  )

# Path towards the data directory
DATA_FOLDER_PATH = Path("data2")

# Path towards the file where all the calculated data is saved
SAVED_DATA_PATH = Path("bovw.joblib") 

# Batch size of MiniBatchKmeans (this value is only used when the dataset is 
# larger than the given number)
BATCH_SIZE = 50000

# Look for the best number of clusters between these ranges
MIN_NUM_CLUSTERS = 10
MAX_NUM_CLUSTERS = 40

# Number of clusters to test from the valid range of clusters defined
NUM_CLUSTERS_TO_TEST = 10

# Size of the thumbnails returnd by the server
THUMBNAIL_SIZE = 256

RESIZE_WIDTH = 500

# Feature extractor
extractor = cv2.BRISK_create(thresh=30)