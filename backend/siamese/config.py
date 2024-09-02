"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# Standard Library imports
from pathlib import Path
import re

# External imports
import tensorflow as tf


def extract_epoch_and_loss(filename: str | Path | None):
    """
    Return
        initial epoch, initial loss threshold
    """

    if filename is None:
        return 0, None

    if isinstance(filename, Path):
        filename = str(filename)

    match = re.search(r"epoch_(\d+)-loss_(\d+\.\d+)", filename)
    if match:
        return int(match[1]), float(match[2])
    raise ValueError(f"Incorrect filename format in file '{filename}'")


def get_latest_epoch_filename(folder_path: Path):
    """ """
    latest_epoch = -1
    latest_filename = None

    for filename in folder_path.rglob("*.keras"):
        epoch, _ = extract_epoch_and_loss(filename.name)
        if epoch > latest_epoch:
            latest_filename = filename.name

    return latest_filename


def get_model_path():
    if CKPT_FILENAME is not None:
        return OUTPUT_PATH / CKPT_FILENAME
    return None


########################################################################################################################
#  Input paths
########################################################################################################################

ROOT = Path("/content")
DATA = ROOT / "oracle-cards"
TRAIN_DATASET = DATA / "train"
VALID_DATASET = DATA / "val"
DATA_SUBSET = ROOT / "oracle-cards-subset"
QUERY_DATASET = ROOT / "query"


########################################################################################################################
#  Output paths
########################################################################################################################

OUTPUT_PATH = Path("siamese_output", "densenet121")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Filename of the checkpoint with the largest epoch number
CKPT_FILENAME = get_latest_epoch_filename(OUTPUT_PATH)

# Used for loading, if exists
LOAD_MODEL_PATH = get_model_path()

# New checkpoints will be saved to here
MODEL_CKPT_PATH = OUTPUT_PATH / "epoch_{epoch:02d}-loss_{val_loss:.4f}.keras"

FAISS_INDEX_PATH = OUTPUT_PATH / "index.faiss"
MANUAL_INDEX_PATH = OUTPUT_PATH / "index.pickle"
LOGS_PATH = OUTPUT_PATH / "logs"
IMAGES_DF_PATH = OUTPUT_PATH / "images.csv"


########################################################################################################################
#  Other parameters
########################################################################################################################

# Model input image size
IMAGE_SIZE = (357, 256)

AUTO = tf.data.AUTOTUNE

# Number of features of the embedding generated by the backbone
EMBEDDING_SHAPE = 128  # Densenet121

# Inference parameters
N_RESULTS = 9

# Index parameters
INDEX_TYPE = "dict"  # faiss, dict

EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")


########################################################################################################################
# Training parameters
########################################################################################################################

TRAIN_BACKBONE = False
LEARNING_RATE = 1e-4
INITIAL_EPOCH, _ = extract_epoch_and_loss(CKPT_FILENAME)
INITIAL_LOSS = None  # None to save models starting from any loss
EPOCHS = 100
BATCH_SIZE = 4
NUM_TRAIN_SAMPLES = len(list(TRAIN_DATASET.rglob("*.jpg")))
NUM_VALIDATION_SAMPLES = len(list(VALID_DATASET.rglob("*.jpg")))
STEPS_PER_EPOCH = int(NUM_TRAIN_SAMPLES / BATCH_SIZE)
VALIDATION_STEPS = int(NUM_VALIDATION_SAMPLES / BATCH_SIZE)
