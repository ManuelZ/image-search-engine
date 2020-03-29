# Built-in imports
from pathlib import Path

# Path towards the data directory
DATA_FOLDER_PATH = Path("data2")

# Path towards the file where all the calculated data is saved
SAVED_DATA_PATH = Path("bovw2.joblib") 

# Number of clusters
K = 20

# Batch size of MiniBatchKmeans
BATCH_SIZE = 50000