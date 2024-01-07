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
from bag_of_visual_words import extract_bovw_features, train_bag_of_visual_words
from config import Config
from descriptors import DESCRIPTORS, Describer


def get_images_descriptions(
    describer: Describer, descriptions_path: Path, retrain: bool = False
) -> dict[str, list[np.ndarray]]:
    """
    Feature extraction
    """

    if descriptions_path.exists() and not retrain:
        logging.info("Loading descriptions from local file.")
        (descriptions_dict,) = joblib.load(str(descriptions_path))

    else:
        logging.info("Recalculating descriptions.")
        if config.MULTIPROCESS:
            descriptions_dict = describer.multiprocessed_descriptors_extraction(
                images_paths, n_jobs=config.N_JOBS
            )
        else:
            descriptions_dict = describer.generate_descriptions(images_paths)

        # Descriptions are not really needed, but helps saving them while developing
        joblib.dump((descriptions_dict,), str(descriptions_path), compress=3)

    return descriptions_dict


def main():
    """
    Extract image features from all the images found in `config.DATA_FOLDER_PATH`.
    """

    ###########################################################################
    # Extract features from images
    ###########################################################################

    features = []

    # describer = Describer(DESCRIPTORS)
    # descriptions_dict = get_images_descriptions(describer, config.DESCRIPTIONS_PATH)
    descriptions_dict = {}

    # Extract Bag Of Visual Words features
    clusterer, codebook, descriptions = train_bag_of_visual_words(images_paths)
    bovw_histogram, pipeline = extract_bovw_features(descriptions, codebook, clusterer)
    print(f"bovw_histogram.shape: {bovw_histogram.shape}")
    assert bovw_histogram.shape[0] == len(images_paths)
    features.append(bovw_histogram)

    ###########################################################################
    # Concatenate all the features obtained from one image
    ###########################################################################

    for descriptor_name, descriptions in descriptions_dict.items():
        logging.info(f"Using descriptor '{descriptor_name}'")
        # descriptions: list of arrays of size (n,136)
        descriptions = np.array(descriptions).reshape(len(images_paths), -1)
        assert descriptions.shape[0] == len(images_paths)
        features.append(descriptions)

    images_features = np.concatenate(features, axis=1)

    logging.info(f"Final shape of feature vector: {images_features.shape}.")
    logging.info(
        f"Proportion of zeros in the feature vector: {(images_features < 01e-9).sum() / images_features.size:.3f}."
    )

    # to_save = {
    #     'images_paths': images_paths,
    #     'images_features': images_features,
    # }

    # if "corners" in descriptions_dict:
    #     if 'faiss' in str(clusterer.__class__):
    #         pass
    #     if 'sklearn' in str(clusterer.__class__):
    #         pass

    #     to_save['codebook'] = codebook
    #     to_save['pipeline'] = pipeline

    # if isinstance(clusterer, faiss.Kmeans):
    #     faiss.write_index(index, str())
    # elif isinstance(clusterer, MiniBatchKMeans):
    #     to_save['clusterer'] = clusterer
    #     joblib.dump(to_save, str(config.BOVW_PATH), compress=3)

    logging.info("Done")


if __name__ == "__main__":
    config = Config()

    logging.basicConfig(format=config.LOGGING_FORMAT, level=config.LOGGING_LEVEL)

    images_paths = []
    for ext in config.EXTENSIONS:
        images_paths.extend(config.DATA_FOLDER_PATH.rglob(ext))

    main()
