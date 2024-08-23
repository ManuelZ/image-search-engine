"""
Modified from the Pyimagesearch 5-part series on Siamese networks: https://pyimg.co/dq1w5
"""

# Standard Library imports
from pathlib import Path

# External imports
import tensorflow as tf


# Path to training and validation data
TRAIN_DATASET = Path(<<FILL ME>>)
VALID_DATASET = Path(<<FILL ME>>)

# model input image size
IMAGE_SIZE = (680, 488)

BATCH_SIZE = 4

AUTO = tf.data.AUTOTUNE

# Training parameters
LEARNING_RATE = 0.0001
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 10
EPOCHS = 10

# Path to save the model
OUTPUT_PATH = Path("siamese_output")
MODEL_PATH = OUTPUT_PATH / "siamese_network.keras"
OUTPUT_IMAGE_PATH = OUTPUT_PATH / "output_image.png"
