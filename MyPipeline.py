"""
NOT USED
"""
import pandas as pd
import h5py
import cv2
import numpy as np
from pathlib import Path
import time
from config import BATCH_SIZE
from config import K
from sklearn.cluster import MiniBatchKMeans

extractor = cv2.BRISK_create()

# This pipeline seems to be fine to pre-process images. Nothing more.

class DatasetGenerator:
    """
    Generate batches of images from given images paths. 
    The idea is that the batch should comfortably fit in memory.
    """
    def __init__(self, image_paths, batch_size=100):
        # Paths towards the images
        self.image_paths = list(image_paths)
        # Number of images to process at a time
        self.batch_size = batch_size
        # Total number of images in the dataset
        self.n_images = len(self.image_paths)

    def generator(self, passes=np.inf):
        """
        Yield a batch of images.
        """
        start = 0
        limits = np.linspace(self.batch_size, self.n_images, self.n_images//self.batch_size, dtype=int)
        for end in limits:
            images = []
            for image_path in self.image_paths[start:end]:
                image = cv2.imread(str(image_path))
                images.append(image)
                start = end
            yield images
            del images


class HDF5DatasetGenerator:
    """
    Modified from: Pyimagesearch DL4CV
    """
    def __init__(self, dbPath, batch_size):
        self.batch_size = batch_size
        self.db = h5py.File(dbPath, "r")
        self.n_images = self.db["labels"].shape[0]

    def generator(self, passes=np.inf) -> list:
        epochs = 0
        while epochs < passes:
            # Loop over the HDF5 dataset
            for i in np.arange(0, self.n_images, self.batch_size):
                images = self.db["images"][i: i + self.batch_size]
                yield images
            epochs += 1

    def close(self):
        self.db.close()


class ImagesFeatureExtractorPipeline:
    """
    Given an image generator, call the 'transform' method of each of the passed processors .
    """
    def __init__(self, steps):
        self.steps = steps
    
    def transform(self, X) -> np.ndarray:
        """
        Args:
            X (generator): Iterable of images

        Returns:
            np.array: Stacked features from all the dataset.
        """
        batches_results = []
        for i,batch in enumerate(X):
            print(f'Working on batch {i+1}...')
            for step in self.steps:
                batch = step[1].transform(batch)
            batches_results.append(batch)
        stacked = np.concatenate([batch_result for batch_result in batches_results], axis=0)
        return stacked


class GrayTransformer:
    """
    Transform images into grayscale.
    """
    def transform(self, X: list) -> list:
        """
        Args:
            X (list): Images
        
        Returns:
            list: Gray images
        """
        return [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in X]


class FeatureDetectorDescriptorTransformer:
    """
    Extract features and descriptors from images.
    """
    def transform(self, X: list) -> np.ndarray:
        """
        Args:
            X (list): Images in grayscale
        Returns:
            np.array: Stacked descriptors of all the images
        """
        results = []
        for image in X:
            kp,des = extractor.detectAndCompute(image, None)
            results.append(des)
        stacked = np.concatenate([im_result for im_result in results], axis=0)
        return stacked


class Quantizer:
    #
    # TODO: this should not be used in this pipeline, 
    # this pipeline applies all the processers in each batch, but this
    # processer requires to have all the features dataset complete
    #

    def __init__(self, batch_size=500):
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(n_clusters=K,
                             random_state=0,
                             max_iter=500,
                             batch_size=batch_size,
                             verbose=0,
                             max_no_improvement=20,
                             )
    def transform(self, X):
        # Number of iterations over the whole dataset
        n_passes = 100
        # Max number of consecutive passes over the whole dataset without improvement
        max_passes_no_improvement = 3
        # Store number of failed passes over the whole dataset
        failed_passes = 0
        # Total number of descriptors in the whole dataset
        n_des = X.shape[0]
        # Inertia of the last run
        old_inertia = 0

        # 3 values average (1 / (1-beta))
        beta = 0.6666

        for i in range(n_passes):
            np.random.shuffle(X)
            start = 0
            limits = np.linspace(self.batch_size, n_des, n_des//self.batch_size, dtype=int)
            print(f'Pass {i+1} over the dataset.')
            for end in limits:
                self.kmeans.partial_fit(X[start:end])
                start += self.batch_size
            
            inertia = self.kmeans.inertia_
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
        codebook = self.kmeans.cluster_centers_
        
        quantized_stacked = np.zeros(X.shape)
        print('Building histograms...')
        for i, (im_path, img_des) in enumerate(X):
            # Translate a descriptor into a quantized descriptor
            quantized_desc = codebook[kmeans.predict(img_des)]
            quantized_stacked[i] = quantized_desc
        return quantized_stacked


if __name__ == '__main__':

    image_paths = Path("data2").rglob("*.jpg")
    X_gen = DatasetGenerator(image_paths, batch_size=1000).generator()

    pipeline = ImagesFeatureExtractorPipeline([
        ('to_gray', GrayTransformer()),
        ('detect_features', FeatureDetectorDescriptorTransformer())
    ])

    start = time.time()
    results = pipeline.transform(X_gen)
    end = time.time()
    print(f'Took {end - start:.1f} seconds.')
