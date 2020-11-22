# Built-in imports
from pathlib import Path
import time

# External imports
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import joblib
from tqdm import tqdm # Progress bars

# Local imports
from config import DATA_FOLDER_PATH
from config import SAVED_DATA_PATH
from config import K
from config import BATCH_SIZE


"""
Modified from:
    - https://www.youtube.com/watch?v=PRceoMWcv1U&t=22s
    - https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/
    - https://gurus.pyimagesearch.com/the-bag-of-visual-words-model/
"""

extractor = cv2.BRISK_create()
n_files = len(list(DATA_FOLDER_PATH.rglob('*.jpg')))


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

    # Inertia of the last run
    old_inertia = 0

    # Number of failed passes
    failed_passes = 0

    for i in range(MAX_NUM_PASSES):
        print(f'\nPass {i+1} over the dataset.')
        
        np.random.shuffle(stacked_descriptors)
        
        beta = 0.666 # 3 values average (1 / (1-beta))

        limits = np.linspace(batch_size, n_des, n_des//batch_size, dtype=int)
        start = 0
        for end in tqdm(limits):
            kmeans.partial_fit(stacked_descriptors[start:end])
            start += batch_size
        
        inertia = kmeans.inertia_
        inertia_diff = 0 if i == 0 else (inertia - old_inertia)
        tracked_inertia = 0 if i == 0 else (tracked_inertia * beta + (1 - beta) * inertia_diff)
        old_inertia = inertia
        print(f'Inertia: {inertia/1024/1024:.0f} M')
        print(f'Inertia diff: {inertia_diff/1024/1024:.0f} M')
        print(f'Tracked inertia: {tracked_inertia/1024/1024:.0f} M')
        
        if (i > 0):
            if tracked_inertia > 0:
                failed_passes += 1
                print(f"Inertia is increasing, failed passes: {failed_passes}.")
                if failed_passes > MAX_FAILED_PASSES:
                    print(f"Stopping multiple dataset passes because there has been no improvement in {failed_passes} passes.")
                    print(f"Finished with total inertia of {inertia/1024/1024:.0f} M")
                break
            else:
                failed_passes = 0


    # K x N where N is the descriptor size
    codebook = kmeans.cluster_centers_ 
    
    #
    # Histogram
    #
    print('Building histograms...')
    hist_features = np.zeros((n_files, K))
    for i, img_des in tqdm(enumerate(descriptors_original)):
        # Translate a descriptor into a quantized descriptor
        quantized_desc = codebook[kmeans.predict(img_des)]
        values, _ = np.histogram(quantized_desc, bins=K)
        hist_features[i] = values
    
    return (kmeans, hist_features, codebook)


def  main():
    """
    Extract image features from all the images found in `DATA_FOLDER_PATH`.
    """
    images_paths = DATA_FOLDER_PATH.rglob('*.jpg')

    # TODO: Add more features, like texture features:
    # Haralick texture, Local Binary Patterns, and Gabor filters.
    # HOG

    print('Extracting image features...')
    original_descriptors = []
    paths_to_images = []
    for i, img_path in tqdm(enumerate(images_paths)):
        image = cv2.imread(str(img_path))
        kp, des = compute_image_descriptors(image)
        if des is None:
            continue
        original_descriptors.append(des)
        paths_to_images.append(img_path)

    stacked_descriptors = np.concatenate(
        [d for d in original_descriptors], axis=0
    ) 
    
    kmeans, hist_features, codebook = create_histogram(
        stacked_descriptors, original_descriptors)

    pipeline = Pipeline([
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('scaler', StandardScaler(with_mean=False)),
    ])
    tfidf_hist_scaled_features = pipeline.fit_transform(hist_features)

    # Not used currently
    # neighbors = NearestNeighbors(n_neighbors=5).fit(tfidf_hist_scaled_features)

    joblib.dump((
        kmeans,
        K,
        pipeline.named_steps['tfidf'],
        pipeline.named_steps['scaler'],
        # neighbors,
        codebook,
        tfidf_hist_scaled_features,
        paths_to_images), str(SAVED_DATA_PATH), compress=3)
    
    print("Done")


if __name__ == "__main__":
    main()