"""
Bag of Visual Words with
    - Brisk descriptors
    - HSV color features

References:
    - http://vision.stanford.edu/teaching/cs131_fall1718/files/14_BoW_bayes.pdf
    - http://vision.stanford.edu/teaching/cs131_fall1718/files/cs131-class-notes.pdf
    - http://www.cs.cmu.edu/~16385/s15/lectures/Lecture12.pdf
    - https://www.youtube.com/watch?v=PRceoMWcv1U&t=22s
    - https://gurus.pyimagesearch.com/the-bag-of-visual-words-model/
"""

# Built-in imports
from pathlib import Path
import time
import random
import logging 

# External imports
import numpy as np
import cv2
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm # Progress bars
from sklearn.metrics import silhouette_score

# Local imports
from descriptors import ColorDescriptor
from config import DATA_FOLDER_PATH
from config import SAVED_DATA_PATH
from config import BATCH_SIZE
from config import EXTENSIONS
from config import MIN_NUM_CLUSTERS
from config import MAX_NUM_CLUSTERS
from config import NUM_CLUSTERS_TO_TEST
from config import RESIZE_WIDTH
from config import extractor
from config import LOGGING_LEVEL
from config import LOGGING_FORMAT
from config import SILHOUETTE_SAMPLE_SIZE
from config import SILHOUETTE_N_SAMPLES

logging.basicConfig(format=LOGGING_FORMAT, level=LOGGING_LEVEL)


images_paths = []
for ext in EXTENSIONS:
    images_paths.extend(DATA_FOLDER_PATH.rglob(ext))
n_files = len(images_paths)

def resize(im, target_width, ratio=None):
    """ Resize an image to attain the target size """
    if not ratio:
        ratio = target_width / im.shape[1]
    dim = (int(target_width), int(im.shape[0] * ratio))
    return cv2.resize(im, dim, interpolation = cv2.INTER_LINEAR_EXACT)


def compute_image_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    kp, des = extractor.detectAndCompute(gray, None)
    return kp, des


def batched_kmeans(stacked_descriptors, kmeans, batch_size):
    """
    Args:
        stacked_descriptors
        kmeans
        batch_size
    """

    n_des = stacked_descriptors.shape[0]
    
    np.random.shuffle(stacked_descriptors)

    start = 0
    limits = np.linspace(batch_size, n_des, n_des//batch_size, dtype=int)
    for end in tqdm(limits):
        kmeans.partial_fit(stacked_descriptors[start:end])
        start = end
    

def create_codebook(original_descriptors):
    """
    Create a dictionary of visual words (visual vocabulary, codebook) out of 
    the given descriptors. Instead of having an infinite number of possible
    points in the space, reduce the possibilities to a certain fixed number of
    clusters.
    
    How to choose vocabulary size (number of clusters)? 
    - Too small: visual words not representative of all patches
    - Too large: quantization artifacts, overfitting
    --> Silhouette coeffiecient

    Args
        original_descriptors: 
    """

    stacked_descriptors = np.concatenate(
        [d for d in original_descriptors], axis=0
    ) 

    # Batch size >= number of extracted descriptors
    batch_size = np.min([stacked_descriptors.shape[0], BATCH_SIZE])

    # Split the available range for the number of clusters
    range_clusters = np.arange(
        MIN_NUM_CLUSTERS,
        MAX_NUM_CLUSTERS + 1,
        (MAX_NUM_CLUSTERS - MIN_NUM_CLUSTERS) / (NUM_CLUSTERS_TO_TEST - 1),
        dtype=np.int
    )

    best_clusterer        = None
    best_n_clusters       = None
    best_silhouette_score = -float("inf")
    
    for n_clusters in range_clusters:

        kmeans = MiniBatchKMeans(
            n_clusters         = n_clusters,
            random_state       = 0,
            max_iter           = 500,
            batch_size         = batch_size,
            verbose            = 0,
            tol                = 1, # tune this
            max_no_improvement = 5,
        )

        if batch_size < BATCH_SIZE:
            logging.info(f"Running Kmeans with {n_clusters} clusters...")
            kmeans.partial_fit(stacked_descriptors)
        else:
            logging.info(f"Running Batch Kmeans with {n_clusters} clusters...")
            batched_kmeans(stacked_descriptors, kmeans, batch_size)
        
        logging.info("Finished running Kmeans.")
        
        s = calculate_sampled_silhouette(kmeans, stacked_descriptors, batch_size)

        logging.info(f"Mean Silhouette score for {n_clusters} clusters: {np.mean(s):.3f} ± {np.std(s):.4f}")
        logging.info(f"Total inertia of {kmeans.inertia_/1024/1024:.0f} M")

        if np.mean(s) > best_silhouette_score:
            best_clusterer        = kmeans
            best_n_clusters       = n_clusters
            best_silhouette_score = np.mean(s)
    

    logging.info(f"Kmeans selected number of clusters: {best_n_clusters}.")
    # Coordinates of cluster centers. 
    # n_clusters x N where N is the descriptor size
    codebook = best_clusterer.cluster_centers_ 
    
    return best_clusterer, best_n_clusters, codebook


def calculate_sampled_silhouette(kmeans, stacked_descriptors, batch_size):
    """
    Calculate the Silhouette score of this KMeans trained instance in a sample
    of observations, multiple times. 
    """

    # TODO: Predict only for the sample
    logging.info(f'Calculating average sampled Silhouette score...')
    
    cluster_labels = []
    n_des = stacked_descriptors.shape[0]
    limits = np.linspace(batch_size, n_des, n_des//batch_size, dtype=int)
    start = 0
    for end in tqdm(limits):
        cluster_labels.extend(kmeans.predict(stacked_descriptors[start:end]))
        start = end

    # ndarray of shape (n_samples, )
    cluster_labels = np.array(cluster_labels)

    # Sample because calculating the Silhouette score takes too long with Brisk
    s = []
    for _ in range(SILHOUETTE_N_SAMPLES):
        sample_idxs = random.sample(
            range(stacked_descriptors.shape[0]),
            SILHOUETTE_SAMPLE_SIZE
        )
        stacked_descriptors_sample = stacked_descriptors[sample_idxs, :]
        cluster_labels_sample = cluster_labels[sample_idxs]
        silhouette_avg = silhouette_score(
            stacked_descriptors_sample,
            cluster_labels_sample
        )
        s.append(silhouette_avg)
    
    return s


def extract_descriptors():
    logging.info(f'Extracting features from {n_files} images...')

    cd = ColorDescriptor((8, 12, 3))
    
    descriptors = []
    images_color_features = []
    for i, img_path in tqdm(enumerate(images_paths)):
        image = cv2.imread(str(img_path))
        if image is None:
            logging.debug("Passing from none image")
            continue
        image = resize(image, RESIZE_WIDTH)
        kp, des = compute_image_descriptors(image)
        if des is None:
            logging.debug("Passing from no descriptors image")
            continue
        descriptors.append(des)
        color_features = cd.describe(image)
        images_color_features.append(color_features)
        
    logging.info("Finished extracting images' features.")
    
    return descriptors, np.array(images_color_features)


def  main():
    """
    Extract image features from all the images found in `DATA_FOLDER_PATH`.
    """

    # TODO: Add more features, like texture features:
    # Haralick texture, Local Binary Patterns, and Gabor filters.
    # HOG

    # TODO: pass a list of feature extractors to this function
    # Step 1: Feature extraction
    # Each image will have multiple feature vectors
    descriptors, images_color_features = extract_descriptors()

    # Step 2: Vocabulary construction (clustering)
    # The cluster centroids (codebook) are the dictionary of visual words
    kmeans, n_clusters, codebook = create_codebook(descriptors)

   # Step 3: Image modelling
   # Each image is modelled as a histogram which tracks the frequency of 
   # clusters (i.e. visual word, centroid). 
   # 1) For each feature vector of an image, find the cluster index to which it
   #    belongs.
   # 2) Create a histogram where the frequency of each cluster index is tracked
    clusters_histograms = np.zeros((len(descriptors), n_clusters))
    for i, des in enumerate(descriptors):
        clusters_idxs = kmeans.predict(des)
        values, _ = np.histogram(clusters_idxs, bins=n_clusters)
        clusters_histograms[i] = values
        logging.debug(f"Information of a new image:")
        logging.debug(f"Clusters indexes: \n{clusters_idxs}")
        logging.debug(f"Clusters histogram: \n{clusters_histograms[i]}")

    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),

        # Transform a count matrix to a normalized tf or tf-idf representation
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
    ])

    logging.info(f"clusters_histograms: {clusters_histograms.shape}")
    clusters_histograms_scaled = pipeline.fit_transform(clusters_histograms)
    logging.info(f"clusters_histograms_scaled: {clusters_histograms_scaled.shape}")
    logging.info(f"images_color_features: {images_color_features.shape}")


    images_features = np.concatenate([clusters_histograms_scaled.todense(), images_color_features], axis=1)

    logging.info(f"Histogram shape: {clusters_histograms_scaled.shape}")
    logging.info(f"Color features shape: {images_color_features.shape}")
    logging.info(f"Final shape: {images_features.shape}")

    joblib.dump((
        kmeans,
        n_clusters,
        codebook,
        pipeline.named_steps['scaler'],
        pipeline.named_steps['tfidf'],
        images_features,
        images_paths
        ), str(SAVED_DATA_PATH), compress=3)
    
    logging.info("Done")


if __name__ == "__main__":
    main()