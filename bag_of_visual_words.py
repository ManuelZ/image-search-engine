"""
Bag of Visual Words with
    - Brisk descriptors
    - HSV color features

References:
    - http://www.cs.cmu.edu/~16385/s15/lectures/Lecture12.pdf
    - https://www.youtube.com/watch?v=PRceoMWcv1U&t=22s
    - https://gurus.pyimagesearch.com/the-bag-of-visual-words-model/
"""

# Built-in imports
from pathlib import Path
import time
import random

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

    # Total number of descriptors in the whole dataset
    n_des = stacked_descriptors.shape[0]
    
    np.random.shuffle(stacked_descriptors)
    
    beta = 0.666 # 3 values average (1 / (1-beta))

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

    print(f"Shape of one descriptor: {original_descriptors[0].shape}")

    stacked_descriptors = np.concatenate(
        [d for d in original_descriptors], axis=0
    ) 

    # Adjust the batch size if the requested one is larger than the available 
    # number of images
    batch_size = np.min([stacked_descriptors.shape[0], BATCH_SIZE])

    range_clusters = np.arange(
        MIN_NUM_CLUSTERS,
        MAX_NUM_CLUSTERS+1,
        (MAX_NUM_CLUSTERS-MIN_NUM_CLUSTERS) / (NUM_CLUSTERS_TO_TEST-1),
        dtype=np.int
    )

    silhouette_scores = []

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

        # just run once if there is no need to run multiple times
        if batch_size < BATCH_SIZE:
            print(f"Running Kmeans with {n_clusters} clusters...")
            kmeans.partial_fit((stacked_descriptors))
            print("Finished running Kmeans")

        else:
            print(f"Running Batch Kmeans with {n_clusters} clusters...")
            batched_kmeans(stacked_descriptors, kmeans, batch_size)
            print("Finished running Batch Kmeans")
        
        s = calculate_sampled_silhouette(kmeans, stacked_descriptors, batch_size)

        silhouette_scores.append(np.mean(s))
        print(f"Mean Silhouette score for {n_clusters} clusters: {np.mean(s):.3f} ± {np.std(s):.4f}")
        print(f"Total inertia of {kmeans.inertia_/1024/1024:.0f} M")
        print()

    idxbest = np.argmax(silhouette_scores)
    best_n_clusters = range_clusters[idxbest]

    kmeans = MiniBatchKMeans(
        n_clusters         = best_n_clusters,
        random_state       = 0,
        max_iter           = 300,
        batch_size         = batch_size,
        verbose            = 0,
        max_no_improvement = 5,
    )

    # just run once if there is no need to run multiple times
    if batch_size < BATCH_SIZE:
        print(f"Final run of Kmeans with {best_n_clusters} clusters with Silhouette score of {silhouette_scores[idxbest]:.4f}.")
        kmeans.partial_fit((stacked_descriptors))
        print("Finished running Kmeans")

    else:
        print(f"Final run of Batch Kmeans with {best_n_clusters} clusters with Silhouette score of {silhouette_scores[idxbest]:.4f}.")
        batched_kmeans(stacked_descriptors, kmeans, batch_size)
        print("Finished running Batch Kmeans")
    
    # Coordinates of cluster centers. 
    # n_clusters x N where N is the descriptor size
    codebook = kmeans.cluster_centers_ 
    
    return kmeans, n_clusters, codebook


def calculate_sampled_silhouette(kmeans, stacked_descriptors, batch_size):
    print(f'Calculating average Silhouette score...')
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
    for _ in range(10):
        sample_idxs = random.sample(range(stacked_descriptors.shape[0]), 5000)
        sample_stacked_descriptors = stacked_descriptors[sample_idxs, :]
        sample_cluster_labels = cluster_labels[sample_idxs]
        silhouette_avg = silhouette_score(sample_stacked_descriptors, sample_cluster_labels)
        s.append(silhouette_avg)
    
    return s


def extract_descriptors():
    print(f'Extracting features from {n_files} images...')

    cd = ColorDescriptor((8, 12, 3))
    
    descriptors = []
    images_color_features = []
    for i, img_path in tqdm(enumerate(images_paths)):
        image = cv2.imread(str(img_path))
        image = resize(image, RESIZE_WIDTH)
        kp, des = compute_image_descriptors(image)
        if des is None:
            continue
        descriptors.append(des)
        color_features = cd.describe(image)
        images_color_features.append(color_features)
    print('Finished extracting image features.')
    
    return descriptors, np.array(images_color_features)


def  main():
    """
    Extract image features from all the images found in `DATA_FOLDER_PATH`.
    """

    # TODO: Add more features, like texture features:
    # Haralick texture, Local Binary Patterns, and Gabor filters.
    # HOG

    # Step 1: Feature extraction
    # Each image will have multiple feature vectors
    descriptors, images_color_features = extract_descriptors()

    # Step 2: Vocabulary construction (clustering)
    # The cluster centroids (codebook) are the dictionary of visual words
    kmeans, n_clusters, codebook = create_codebook(descriptors)

   # Step 3: Image modelling
   # Each image is modelled as a histogram which tracks the frequency of each 
   # quantized feature vector (i.e. visual word, centroid). 
   # 1) Each feature vector of an image is quantized (i.e. for a feture vector 
   #    look into the codebook for the closest centroid).
   # 2) Create a histogram out of all the quantized feature vectors of the 
   #    image.
    hist_features = np.zeros((n_files, n_clusters))
    for i, des in enumerate(descriptors):
        quantized_desc = codebook[kmeans.predict(des)]
        values, _ = np.histogram(quantized_desc, bins=n_clusters)
        hist_features[i] = values

    pipeline = Pipeline([
        # ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('scaler', StandardScaler(with_mean=False)),
    ])
    hist_features_scaled = pipeline.fit_transform(hist_features)
    images_features = np.concatenate([hist_features_scaled, images_color_features], axis=1)

    print(f"Histogram shape: {hist_features_scaled.shape}")
    print(f"Color features shape: {images_color_features.shape}")
    print(f"Final shape: {images_features.shape}")

    joblib.dump((
        kmeans,
        n_clusters,
        # pipeline.named_steps['tfidf'],
        pipeline.named_steps['scaler'],
        codebook,
        images_features,
        images_paths
        ), str(SAVED_DATA_PATH), compress=3)
    
    print("Done")


if __name__ == "__main__":
    main()