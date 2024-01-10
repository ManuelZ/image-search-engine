# TODO: Add more features:
# - Haralick texture
# - Local Binary Patterns
# - Gabor filters
# - sklearn.feature_extraction.image.extract_patches_2d
# - HOG
#
# Also, do an inverted index file to hold the mapping of words to documents
# to quickly compute the similarity between a new image and all of the
# images in the database.


# Standard Library imports
import logging

# External imports
import numpy as np
import numpy.typing as npt

# Local imports
from config import Config
from utils import get_images_paths
from bag_of_visual_words import (
    train_bovw_model,
)
from descriptors import Describer, CornerDescriptor, describe_dataset


def main(images_paths):
    """ """
    ###########################################################################
    # Pre-calculate and save corner descriptions so that they can be loaded in
    # each parallel Cross-Validation process.
    ###########################################################################
    describer = Describer({"corners": CornerDescriptor(config.CORNER_DESCRIPTOR)})
    images_paths = np.array(images_paths).reshape(-1, 1)
    descriptions = describe_dataset(describer, images_paths)

    print(
        f"Mean number of descriptions per image: {np.mean([d.shape[0]  for d in descriptions[:100]])}"
    )

    ###########################################################################
    # Extract BOVW features
    ###########################################################################
    train_bovw_model(images_paths, describer)


if __name__ == "__main__":
    config = Config()

    logging.basicConfig(format=config.LOGGING_FORMAT, level=config.LOGGING_LEVEL)

    images_paths = get_images_paths()

    main(images_paths)
