# Built-in imports
from pathlib import Path
import time

# External imports
from sklearn.cluster import MiniBatchKMeans
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import cv2
import joblib
from tqdm import tqdm # Progress bars

# Local imports
from config import DATA_FOLDER_PATH
from config import SAVED_DATA_PATH
from config import K
from config import BATCH_SIZE
from MyPipeline import ImagesFeatureExtractorPipeline
from MyPipeline import DatasetGenerator
from MyPipeline import GrayTransformer
from MyPipeline import FeatureDetectorDescriptorTransformer

"""
Modified from:
    - https://www.youtube.com/watch?v=PRceoMWcv1U&t=22s
    - https://www.pyimagesearch.com/2014/12/01/
    - complete-guide-building-image-search-engine-python-opencv/
    - https://gurus.pyimagesearch.com/the-bag-of-visual-words-model/
"""

extractor = cv2.BRISK_create()
images_paths = DATA_FOLDER_PATH.rglob('*.jpg')
n_files = len(list(DATA_FOLDER_PATH.rglob('*.jpg')))


def create_bag_of_descriptors(descriptors_original):
    """ 
    Stack all the given into a single Numpy array.
    """
    print('Creating bag of descriptors...')

    img_path, descriptors_stacked = descriptors_original[0]

    descriptors_stacked = np.concatenate([d[1] for d in tqdm(descriptors_original)], axis=0)
    return descriptors_stacked


def compute_image_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    kp, des = extractor.detectAndCompute(gray, None)
    return kp, des


def chi2_distance(histA, histB, eps=1e-10):
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
    return d


def  main():

    # TODO: Add more features, like texture features:
    # Haralick texture, Local Binary Patterns, and Gabor filters.
    # HOG

    print('Extracting image features...')
    descriptors_original = []
    paths_to_images = []
    for i, img_path in tqdm(enumerate(images_paths)):
        image = cv2.imread(str(img_path))
        kp, des = compute_image_descriptors(image)
        if des is None:
            continue
        descriptors_original.append((img_path, des))
        paths_to_images.append(img_path)

    descriptors_stacked = create_bag_of_descriptors(descriptors_original)

    # Verify a correct batch size
    batch_size = np.min([descriptors_stacked.shape[0], BATCH_SIZE])
    
    kmeans = MiniBatchKMeans(n_clusters=K,
                             random_state=0,
                             max_iter=500,
                             batch_size=batch_size,
                             verbose=0,
                             max_no_improvement=20,
                             )
    
    # Number of iterations over the whole dataset
    n_passes = 100
    # Max number of consecutive passes over the whole dataset without improvement
    max_passes_no_improvement = 3
    # Store number of failed passes over the whole dataset
    failed_passes = 0
    # Total number of descriptors in the whole dataset
    n_des = descriptors_stacked.shape[0]
    # Inertia of the last run
    old_inertia = 0

    # 3 values average (1 / (1-beta))
    beta = 0.6666

    for i in range(n_passes):
        np.random.shuffle(descriptors_stacked)
        start = 0
        limits = np.linspace(batch_size, n_des, n_des//batch_size, dtype=int)
        print(f'Pass {i+1} over the dataset.')
        for end in tqdm(limits):
            kmeans.partial_fit(descriptors_stacked[start:end])
            start += batch_size
        
        inertia = kmeans.inertia_
        inertia_diff = old_inertia - inertia        

        if i == 0:
            tracked_inertia = inertia
        elif (i > 0):
            if tracked_inertia < 0:
                print("Early stopping because there is no improvement.")
                break
        
        tracked_inertia = tracked_inertia * beta + (1 - beta) * inertia_diff
        old_inertia = inertia
        print(f'Inertia: {inertia:.0f}')
        print(f'Inertia diff: {inertia_diff:.0f}')
        print(f'Tracked inertia: {tracked_inertia:.0f}\n')

    # K x N where N is the descriptor size
    codebook = kmeans.cluster_centers_ 
    
    #
    # Histogram
    #
    print('Building histograms...')
    hist_features = np.zeros((n_files, K))
    for i, (im_path, img_des) in tqdm(enumerate(descriptors_original)):
        # Translate a descriptor into a quantized descriptor
        quantized_desc = codebook[kmeans.predict(img_des)]
        values, _ = np.histogram(quantized_desc, bins=K)
        hist_features[i] = values


    # TODO: put this in a pipeline

    tfidf = TfidfTransformer(sublinear_tf=True)
    tfidf_hist_features = tfidf.fit_transform(hist_features)

    scaler = StandardScaler(with_mean=False)
    tfidf_hist_scaled_features = scaler.fit_transform(tfidf_hist_features)
    
    neighbors = NearestNeighbors(n_neighbors=5).fit(tfidf_hist_scaled_features)

    joblib.dump((kmeans, K, tfidf, scaler, neighbors, codebook, tfidf_hist_scaled_features, paths_to_images), str(SAVED_DATA_PATH), compress=3)
    print("Done")


if __name__ == "__main__":
    main()