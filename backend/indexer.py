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

# Mean images width and height
# shapes = np.zeros((len(images_paths),2))
# for i,p in tqdm(enumerate(images_paths)):
#     image = io.imread(str(p))
#     h,w,_ = image.shape 
#     shapes[i,0] = h
#     shapes[i,1] = w
# print(f"Mean height: {shapes[:,0].mean():.1f} +- {shapes[:,0].std():.1f}")
# print(f"Mean widtht: {shapes[:,1].mean():.1f} +- {shapes[:,1].std():.1f}")



import logging
import numpy as np
import joblib
from config import Config
from descriptors import (
    DESCRIPTORS,
    Describer
)
from bag_of_visual_words import create_bovw
import faiss
from sklearn.cluster import MiniBatchKMeans

config = Config()

logging.basicConfig(format=config.LOGGING_FORMAT, level=config.LOGGING_LEVEL)

images_paths = []
for ext in config.EXTENSIONS:
    images_paths.extend(config.DATA_FOLDER_PATH.rglob(ext))


def get_images_descriptions(describer: Describer) -> dict[str, list[np.ndarray]]:
    """
    Feature extraction
    """

    if config.DESCRIPTIONS_PATH.exists():
        logging.info("Loading descriptions from local file.")
        descriptions_dict, = joblib.load(str(config.DESCRIPTIONS_PATH))
    
    else:
        logging.info("Recalculating descriptions.")

        # Apparently, SIFT can't get from one processes to another
        if DESCRIPTORS.get('corners') and (DESCRIPTORS.get('corners').kind == 'sift'):
            descriptions_dict = describer.generate_descriptions(images_paths)
        elif config.MULTIPROCESS:
            descriptions_dict = describer.multiprocessed_descriptors_extraction(images_paths, n_jobs=config.N_JOBS)
        else:
            descriptions_dict = describer.generate_descriptions(images_paths)
        
        # Descriptions are not really needed, but helps saving them while developing
        joblib.dump((descriptions_dict,), str(config.DESCRIPTIONS_PATH), compress=3)
    
    return descriptions_dict


def  main():
    """
    Extract image features from all the images found in `config.DATA_FOLDER_PATH`.
    """
    describer = Describer(DESCRIPTORS)

    descriptions_dict = get_images_descriptions(describer)

    ###########################################################################
    # Concatenate features:
    # Concatenate all the features obtained for one image
    ###########################################################################

    features = []
    for descriptor_name, descriptions in descriptions_dict.items():

        # `descriptions` is a list of arrays of size (n,136)
        logging.info(f"Using descriptor '{descriptor_name}'")

        if descriptor_name == 'corners':
            bovw_histogram, clusterer, codebook, pipeline = create_bovw(descriptions)
            logging.info(f"Histogram shape: {bovw_histogram.shape}.")
            assert bovw_histogram.shape[0] == len(images_paths)
            features.append(bovw_histogram)
        else:
            descriptions = np.array(descriptions).reshape(len(images_paths),-1)
            assert descriptions.shape[0] == len(images_paths)
            features.append(descriptions)

    images_features = np.concatenate(features, axis=1)

    logging.info(f"Final shape of feature vector: {images_features.shape}.")
    logging.info(f"Proportion of zeros in the feature vector: {(images_features < 01e-9).sum() / images_features.size:.3f}.")

    to_save = {
        'images_paths': images_paths,
        'images_features': images_features,
    }
    
    if "corners" in descriptions_dict:
        if 'faiss' in str(clusterer.__class__):
            pass
        if 'sklearn' in str(clusterer.__class__):
            pass
        
        to_save['codebook'] = codebook
        to_save['pipeline'] = pipeline

    # if isinstance(clusterer, faiss.Kmeans):
    #     faiss.write_index(index, str())
    # elif isinstance(clusterer, MiniBatchKMeans):
    #     to_save['clusterer'] = clusterer
    #     joblib.dump(to_save, str(config.BOVW_PATH), compress=3)
    
    logging.info("Done")


if __name__ == "__main__":
    main()