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
import multiprocessing as mp

# External imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import joblib
from tqdm import tqdm # Progress bars
import skimage
from skimage import io
from skimage.color import rgb2gray
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.pipeline import Pipeline


# Local imports
from utils import resize
from utils import chunkIt
from utils import FaissKMeans
from descriptors import ColorDescriptor
from descriptors import CornerDescriptor
from descriptors import HOGDescriptor
from config import DESCRIPTORS
from config import DATA_FOLDER_PATH
from config import SAVED_DATA_PATH
from config import BATCH_SIZE
from config import EXTENSIONS
from config import MIN_NUM_CLUSTERS
from config import MAX_NUM_CLUSTERS
from config import NUM_CLUSTERS_TO_TEST
from config import RESIZE_WIDTH
from config import LOGGING_LEVEL
from config import LOGGING_FORMAT
from config import CLUSTER_EVAL_SAMPLE_SIZE
from config import CLUSTER_EVAL_N_SAMPLES
from config import SAVED_DESCRIPTIONS_PATH
from config import CLUSTER_EVAL_METHOD
from config import CLUSTER_RETURN_ALL

logging.basicConfig(format=LOGGING_FORMAT, level=LOGGING_LEVEL)


images_paths = []
for ext in EXTENSIONS:
    images_paths.extend(DATA_FOLDER_PATH.rglob(ext))
    

def create_codebook(original_descriptors):
    """
    Create a dictionary of visual words (visual vocabulary, codebook) out of 
    the given descriptors. Instead of having an infinite number of possible
    points in the space, reduce the possibilities to a certain fixed number of
    clusters.
    
    How to choose vocabulary size (number of clusters)? 
    - Too small: visual words not representative of all patches
    - Too large: quantization artifacts, overfitting

    Args
        original_descriptors: 
    """

    stacked_descriptors = np.concatenate(original_descriptors, axis=0)

    logging.info(f"Number of descriptors in the dataset: {stacked_descriptors.shape[0]}")

    # Batch size >= number of extracted descriptors
    batch_size = np.min([stacked_descriptors.shape[0], BATCH_SIZE])

    max_clusters = np.min([MAX_NUM_CLUSTERS + 1, stacked_descriptors.shape[0]])

    if NUM_CLUSTERS_TO_TEST == 1:
        range_clusters = [MIN_NUM_CLUSTERS]

    else:
        # Split the available range in n clusters
        range_clusters = np.arange(
            MIN_NUM_CLUSTERS,
            max_clusters,
            (max_clusters - MIN_NUM_CLUSTERS) / NUM_CLUSTERS_TO_TEST,
            dtype=np.int
        )

    logging.info(f"Number of clusters to test: {range_clusters}")

    best_clusterer        = None
    best_n_clusters       = None

    if CLUSTER_EVAL_METHOD == 'silhouette':
        best_cluster_score = -float("inf")
    
    elif CLUSTER_EVAL_METHOD == 'davies-bouldin':
        best_cluster_score = float("inf")

    elif  CLUSTER_EVAL_METHOD == 'calinski_harabasz_score':
        best_cluster_score = -float("inf")

    clustering_results = {
        'n_clusters' : [],
        'time'       : [],
        'score'      : [],
        'inertia'    : [],
        'silhouette' : [],
        'davies-bouldin': [],
        'calinski_harabasz_score' : []
    }

    for i,n_clusters in enumerate(range_clusters):

        start = time.time()

        clusterer = MiniBatchKMeans(
            n_clusters         = n_clusters,
            random_state       = 42,
            max_iter           = 100,
            batch_size         = batch_size,
            verbose            = 0,
            tol                = 0, # tune this
            max_no_improvement = 5,
            n_init             = 3
        )

        # clusterer = FaissKMeans(
        #     n_clusters = int(n_clusters),
        #     max_iter   = 60,
        #     n_init     = 2
        # )

        if batch_size < BATCH_SIZE:
            logging.info(f"Running Kmeans with {n_clusters} clusters...")
            clusterer.fit(stacked_descriptors)
        else:
            logging.info(f"Running Batch Kmeans with {n_clusters} clusters...")
            clusterer.fit(stacked_descriptors)
        
        end = time.time()

        logging.info(f"Finished running KMeans. Took {end - start:.1f} sec.")
        
        s = calc_sampled_cluster_score(clusterer, stacked_descriptors)

        logging.info(
            "Mean '{}' score for {} clusters: {:.3f} ± {:.3f}".format(
                s['method'], n_clusters, np.mean(s['scores']), np.std(s['scores'], ddof=1)
            )
        )
        logging.info(f"Total inertia of {clusterer.inertia_/1024:.1f} K")

        if CLUSTER_EVAL_METHOD == 'silhouette':
            if np.mean(s['scores']) > best_cluster_score:
                best_clusterer        = clusterer
                best_n_clusters       = n_clusters
                best_cluster_score = np.mean(s['scores'])
        
        elif CLUSTER_EVAL_METHOD == 'davies-bouldin':
            if np.mean(s['scores']) < best_cluster_score:
                best_clusterer        = clusterer
                best_n_clusters       = n_clusters
                best_cluster_score = np.mean(s['scores'])
        
        elif CLUSTER_EVAL_METHOD == 'calinski_harabasz_score':
            if np.mean(s['scores']) > best_cluster_score:
                best_clusterer        = clusterer
                best_n_clusters       = n_clusters
                best_cluster_score = np.mean(s['scores'])

        # Save information of this run
        clustering_results['n_clusters'].append(n_clusters)
        clustering_results['score'].append(np.mean(s['scores']))
        clustering_results['inertia'].append(clusterer.inertia_)
        clustering_results['time'].append(end-start)

        if CLUSTER_RETURN_ALL:
            clustering_results['silhouette'].append(np.mean(s['all']['silhouette']))
            clustering_results['davies-bouldin'].append(np.mean(s['all']['davies-bouldin']))
            clustering_results['calinski_harabasz_score'].append(np.mean(s['all']['calinski_harabasz_score']))


    if CLUSTER_RETURN_ALL:
        num_graphs = 5
    else:
        num_graphs = 3

    if len(range_clusters) > 1:

        plt.subplot(num_graphs,1,1)
        plt.plot(clustering_results['n_clusters'], clustering_results['time'])
        plt.title("Training time [seconds]")
        
        plt.subplot(num_graphs,1,2)
        plt.plot(clustering_results['n_clusters'], clustering_results['inertia'])
        plt.title("Inertia (lower is better)")

        if not CLUSTER_RETURN_ALL:
            plt.subplot(num_graphs,1,3)
            plt.plot(clustering_results['n_clusters'], clustering_results['score'])
            better = 'lower' if (s['method']=='davies-bouldin') else 'higher'
            plt.title(f"'{s['method']}' ({better} is better)")
        
        else:
            plt.subplot(num_graphs,1,3)
            plt.plot(clustering_results['n_clusters'], clustering_results['davies-bouldin'])
            plt.title(f"Davies-Bouldin (lower is better)")

            plt.subplot(num_graphs,1,4)
            plt.plot(clustering_results['n_clusters'], clustering_results['silhouette'])
            plt.title(f"Silhouette (higher is better)")

            plt.subplot(num_graphs,1,5)
            plt.plot(clustering_results['n_clusters'], clustering_results['calinski_harabasz_score'])
            plt.title(f"Calinski-Harabasz (higher is better)")


        plt.show()

        logging.info(f"Kmeans selected number of clusters: {best_n_clusters}.")
    
    # Coordinates of cluster centers. 
    # n_clusters x N where N is the descriptor size
    codebook = best_clusterer.cluster_centers_ 
    
    return best_clusterer, codebook


def calc_sampled_cluster_score(clusterer, stacked_descriptors):
    """
    Calculate an evaluation score of this KMeans trained instance in a sample
    of observations, multiple times.
    It's neccesary to sample because otherwise the time it takes to compute the
    whole dataset is huge.

    For Silhouette:
        The best value is 1 and the worst value is -1. 
        Values near 0 indicate overlapping clusters. 
        Negative values generally indicate that a sample has been assigned to the 
        wrong cluster, as a different cluster is more similar.

    For Davie-Bouldin:
        Zero is perfect.
        This index signifies the average ‘similarity’ between clusters, where 
        the similarity is a measure that compares the distance between 
        clusters with the size of the clusters themselves.

    For calinski_harabasz_score:
        Higher is better.
        The index is the ratio of the sum of between-clusters dispersion and 
        of inter-cluster dispersion for all clusters 
        (where dispersion is defined as the sum of distances squared).
    """

    dataset_size = stacked_descriptors.shape[0]
    sample_size = np.min([dataset_size, CLUSTER_EVAL_SAMPLE_SIZE])
    logging.info(f"Calculating mean sampled (n={sample_size}) '{CLUSTER_EVAL_METHOD}' score...")

    if CLUSTER_RETURN_ALL:
        all_scores = {
            'silhouette' : [],
            'davies-bouldin': [],
            'calinski_harabasz_score': [],
        }
    scores = []
    
    for _ in range(CLUSTER_EVAL_N_SAMPLES):
        
        sample_idxs = random.sample( range(dataset_size), sample_size)
        sample = stacked_descriptors[sample_idxs, :]
        clst_idx_samp = clusterer.predict(sample)

        if CLUSTER_RETURN_ALL:
            all_scores['silhouette'].append(silhouette_score(sample, clst_idx_samp.ravel()))
            all_scores['davies-bouldin'].append(davies_bouldin_score(sample, clst_idx_samp.ravel()))
            all_scores['calinski_harabasz_score'].append(calinski_harabasz_score(sample, clst_idx_samp.ravel()))

        if CLUSTER_EVAL_METHOD == 'silhouette':
            scorer = silhouette_score
        elif CLUSTER_EVAL_METHOD == 'davies-bouldin':
            scorer = davies_bouldin_score
        elif CLUSTER_EVAL_METHOD == "calinski_harabasz_score":
            scorer = calinski_harabasz_score

        score_sample = scorer(sample, clst_idx_samp.ravel())
        
        scores.append(score_sample)

    return { 'method' : CLUSTER_EVAL_METHOD, 'scores': scores, 'all' : all_scores }


def extract_descriptors(images_paths, descriptors, queue=None):
    """

    Args:
        descriptors: dict of descriptor name and descriptor object with a 
                     .describe() method.
    """

    logging.info(f'Extracting features from {len(images_paths)} images...')

    # Dict of lists where all the images' descriptors will be stored
    extracted = { d_name: [] for d_name,_ in descriptors.items() }

    for i, img_path in tqdm(enumerate(images_paths)):
        image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #image = io.imread(img_path)
        #image = rgb2gray(image)
        image = skimage.transform.resize(image, (RESIZE_WIDTH,RESIZE_WIDTH))
        
        for d_name, descriptor in descriptors.items():
            
            try:
                description = descriptor.describe(image)
            
            except Exception as e:
                description = None
                print(f"Trouble describing image {img_path}\n {e}")
            
            # Brisk sometimes may not find corners
            if description is None:
                continue
            
            extracted[d_name].append(description)

    extracted = { d_name : np.array(l) for d_name,l in extracted.items() }
    
    logging.info("Finished extracting images' features.")
    
    return extracted


def multiprocessed_descriptors_extraction(images_paths, descriptors, n_jobs=4):
    """
    Extract images' descriptions using multiple processes.
    """

    pool = mp.Pool(processes=n_jobs)
    paths_chunks = chunkIt(images_paths, n_jobs)
    
    results = [
        pool.apply_async(
            func     = extract_descriptors,
            args     = (paths, descriptors),
            callback = lambda x: print("Process finished.")
        ) for paths in paths_chunks]
    
    pool.close()
    output = [p.get() for p in results]
    print("All feature extraction processes finished.")

    extracted = { d_name: [] for d_name in descriptors.keys() }

    for r in output:
        for d_name in descriptors.keys():
            extracted[d_name].extend(r[d_name])

    return extracted


def bovw(descriptions):
    """
    Args:
        descriptions: vector of extracted corner features
    """

    ###########################################################################
    # Vocabulary construction (clustering)
    ###########################################################################

    # The cluster centroids (codebook) are the dictionary of visual words
    clusterer, codebook = create_codebook(descriptions)
    n_clusters = clusterer.cluster_centers_.shape[0]


    ###########################################################################
    # Image modeling
    ###########################################################################
    # Each image is modelled as a histogram which tracks the frequency of 
    # clusters (i.e. visual word, centroid). 
    # 1) For each feature vector of an image, find the cluster index to which it
    #    belongs.
    # 2) Create a histogram where the frequency of each cluster index is tracked

    clusters_histograms = np.zeros((len(descriptions), n_clusters))

    for i, des in enumerate(descriptions):
        clusters_idxs = clusterer.predict(des)
        values,_ = np.histogram(clusters_idxs, bins=n_clusters)
        clusters_histograms[i] = values

    pipeline = Pipeline([
        # Histogram normalization. 'norm' selection is dataset-dependent
        # ('normalizer', Normalizer(norm='l1')),
        # Transform a count matrix to a normalized tf or tf-idf representation
        ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
    ])

    bovw_histogram = pipeline.fit_transform(clusters_histograms).todense()

    return bovw_histogram, clusterer, codebook, pipeline


def  main():
    """
    Extract image features from all the images found in `DATA_FOLDER_PATH`.
    """

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

    ###########################################################################
    #  Feature extraction
    ###########################################################################
    # Each image will have multiple feature vectors

    # TODO: if the dataset is too large, extracting all the descriptors at the
    # beginning may collpse the memory. Better to load and clusterize in batches

    if SAVED_DESCRIPTIONS_PATH.exists():
        logging.info("Loading descriptions from local file.")
        descriptions, = joblib.load(str(SAVED_DESCRIPTIONS_PATH))
    
    else:
        logging.info("Recalculating descriptions.")
        # descriptions = extract_descriptors(images_paths, DESCRIPTORS)
        descriptions = multiprocessed_descriptors_extraction(images_paths, DESCRIPTORS, n_jobs=4)
        joblib.dump((descriptions,), str(SAVED_DESCRIPTIONS_PATH), compress=3)


    ###########################################################################
    # Concatenate features
    ###########################################################################

    features = []
    for des_name in descriptions.keys():
        
        if des_name == 'corners':
            bovw_histogram, clusterer, codebook, pipeline = bovw(descriptions['corners'])
            features.append(bovw_histogram)
        else:
            features.append(descriptions[des_name])

    images_features = np.concatenate(features, axis=1)

    if "corners" in descriptions:
        logging.info(f"Histogram shape: {bovw_histogram.shape}.")

    logging.info(f"Final shape of feature vector: {images_features.shape}.")
    logging.info(f"Proportion of zeros in the feature vector: {(images_features < 01e-9).sum() / images_features.size:.3f}.")

    n_clusters = clusterer.cluster_centers_.shape[0]
    to_save = []
    if "corners" in descriptions:
        if 'faiss' in str(clusterer.__class__):
            # write_index(index, "large.index")
            pass
        if 'sklearn' in str(clusterer.__class__):
            pass
            
        to_save.append(clusterer)
        to_save.append(codebook)
        to_save.append(pipeline)
    
    to_save.append(images_features)
    to_save.append(images_paths)
    joblib.dump(to_save, str(SAVED_DATA_PATH), compress=3)
    
    logging.info("Done")


if __name__ == "__main__":
    main()