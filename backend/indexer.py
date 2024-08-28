# Standard Library imports
import logging
from collections import defaultdict
import pickle

# External imports
import numpy as np
import cv2
import faiss
from tqdm import tqdm

# Local imports
from config import Config
from utils import get_images_paths, create_search_index
from bag_of_visual_words import train_bovw_model
from descriptors import (
    Describer,
    CornerDescriptor,
    describe_dataset,
    CNNDescriptor,
    DHashDescriptor,
)

config = Config()


def main():
    """ """
    print("Starting...")
    images_paths = get_images_paths()
    images_paths = np.array(images_paths).reshape(-1, 1)

    if config.METHOD == config.METHOD.BOVW:
        descriptor = CornerDescriptor(config.CORNER_DESCRIPTOR)
        describer = Describer({"corners": descriptor})
        # Extract BOVW features and create index
        train_bovw_model(images_paths, describer)

    elif config.METHOD == config.METHOD.DHASH:
        descriptor = DHashDescriptor()
        describer = Describer({"dhash": descriptor})
        d = defaultdict(list)
        for impath in tqdm(images_paths.ravel()):
            image = cv2.imread(str(impath))
            imhash = descriptor.describe(image)[0][0]
            d[imhash].append(impath)

        with open(config.DHASH_INDEX_PATH, "wb") as f:
            pickle.dump(d, f)

    elif config.METHOD == config.METHOD.DNN:
        descriptor = CNNDescriptor(model=config.DNN_MODEL)
        describer = Describer({"conv_features": descriptor})
        descriptions = describe_dataset(describer, images_paths)
        descriptions = np.concatenate(descriptions)

        print(f"Creating index with features of size ", descriptions.shape)
        index = create_search_index(descriptions, index_type=config.INDEX_TYPE)
        faiss.write_index(index, str(config.DNN_INDEX_PATH))


if __name__ == "__main__":
    logging.basicConfig(format=config.LOGGING_FORMAT, level=config.LOGGING_LEVEL)
    main()

    # TODO: Add more features for the BOVW model:
    # - Haralick texture
    # - Local Binary Patterns
    # - Gabor filters
    # - sklearn.feature_extraction.image.extract_patches_2d
    # - HOG
