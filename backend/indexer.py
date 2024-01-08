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
from pathlib import Path

# External imports
import joblib
import numpy as np
import numpy.typing as npt

# Local imports
from config import Config
from utils import get_images_paths
from bag_of_visual_words import (
    generate_bovw_features_from_descriptions,
    train_bag_of_visual_words_model,
    save_bovw_model,
)
from descriptors import Describer, CornerDescriptor


# TODO: currently it's used for extracting corners, make a different function for that and keep this general
def describe_dataset(
    describer: Describer, images_paths: list[Path]
) -> list[np.ndarray]:
    corner_descriptions_path = config.BOVW_CORNER_DESCRIPTIONS_PATH

    if corner_descriptions_path.exists():
        logging.info("Loading dataset features from local file.")
        descriptions_dict = joblib.load(str(corner_descriptions_path))
    else:
        logging.info("Extracting features from dataset.")
        if config.MULTIPROCESS:
            descriptions_dict = describer.multiprocessed_describe(
                images_paths, n_jobs=config.N_JOBS
            )
        else:
            descriptions_dict = describer.describe(images_paths)

        joblib.dump(descriptions_dict, str(corner_descriptions_path), compress=3)

    descriptions = descriptions_dict["corners"]

    return descriptions


def main():
    """
    Extract image features from all the images found in `config.DATA_FOLDER_PATH`.
    """
    features = []

    ###########################################################################
    # Extract BOVW features
    ###########################################################################

    describer = Describer({"corners": CornerDescriptor(config.CORNER_DESCRIPTOR)})

    # Extract corner features
    descriptions = describe_dataset(describer, images_paths)
    index = train_bag_of_visual_words_model(descriptions)
    bovw_histograms, pipeline = generate_bovw_features_from_descriptions(
        descriptions, index
    )
    assert bovw_histograms.shape[0] == len(images_paths)
    features.append(bovw_histograms)

    np.save(config.BOVW_HISTOGRAMS_PATH, bovw_histograms)

    ###########################################################################
    # Extract other features
    ###########################################################################

    # ...

    ###########################################################################
    # Concatenate all the features obtained from one image
    ###########################################################################

    descriptions_dict = {}
    for descriptor_name, descriptions in descriptions_dict.items():
        logging.info(f"Using descriptor '{descriptor_name}'")
        # descriptions: list of arrays of size (n,136)
        descriptions = np.array(descriptions).reshape(len(images_paths), -1)
        assert descriptions.shape[0] == len(images_paths)
        features.append(descriptions)

    images_features = np.concatenate(features, axis=1)

    logging.info(f"Shape of final feature vector: {images_features.shape}.")
    logging.info(
        f"Proportion of zeros in the feature vector: {(images_features < 01e-9).sum() / images_features.size:.3f}."
    )
    logging.info("Saving Bag of Visual Words model...")
    save_bovw_model(index, pipeline)
    logging.info("Done.")


if __name__ == "__main__":
    config = Config()

    logging.basicConfig(format=config.LOGGING_FORMAT, level=config.LOGGING_LEVEL)

    images_paths = get_images_paths()

    main()
