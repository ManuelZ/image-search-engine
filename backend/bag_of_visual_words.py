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
import time
import random
import logging
from collections import defaultdict

# External imports
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.pipeline import Pipeline

# Local imports
from utils import OkapiTransformer
from config import Config
from descriptors import (
    extract_descriptors,
    multiprocessed_descriptors_extraction,
    DESCRIPTORS
)

config = Config()

logging.basicConfig(format=config.LOGGING_FORMAT, level=config.LOGGING_LEVEL)

images_paths = []
for ext in config.EXTENSIONS:
    images_paths.extend(config.DATA_FOLDER_PATH.rglob(ext))


   
def plot_results(results):
    num_graphs = 5

    ax = plt.subplot(num_graphs,1,1)
    plt.plot(results['n_clusters'], results['time'])
    plt.title("Training time [seconds]")
    
    plt.subplot(num_graphs,1,2, sharex=ax)
    plt.plot(results['n_clusters'], results['inertia'])
    plt.title("Inertia (lower is better)")

    plt.subplot(num_graphs,1,3, sharex=ax)
    plt.plot(results['n_clusters'], results['davies-bouldin'])
    plt.title(f"Davies-Bouldin (lower is better)")

    plt.subplot(num_graphs,1,4, sharex=ax)
    plt.plot(results['n_clusters'], results['silhouette'])
    plt.title(f"Silhouette (higher is better)")

    plt.subplot(num_graphs,1,5, sharex=ax)
    plt.plot(results['n_clusters'], results['calinski-harabasz'])
    plt.title(f"Calinski-Harabasz (higher is better)")

    plt.show()


class ClusteringConfig:
    
    def __init__(self, config, n_descriptors):
        self.config = config
        self.batch_size = self.calc_batch_size(n_descriptors)
        self.max_clusters = np.min([self.config.MAX_NUM_CLUSTERS + 1, n_descriptors])
        self.range_clusters = self.calc_range_clusters()


    def calc_batch_size(self, n_descriptors):
        return np.min([n_descriptors, self.config.BATCH_SIZE])


    def calc_range_clusters(self):
        if self.config.NUM_CLUSTERS_TO_TEST == 1:
            clusters_range = [self.config.MIN_NUM_CLUSTERS]

        else:
            # Split the available range in n clusters
            clusters_range = np.arange(
                self.config.MIN_NUM_CLUSTERS,
                self.max_clusters,
                (self.max_clusters - self.config.MIN_NUM_CLUSTERS) / self.config.NUM_CLUSTERS_TO_TEST,
                dtype=np.int
            )
        
        logging.info(f"Number of clusters to test: {clusters_range}")
        
        return clusters_range


class ClusterEvaluator:

    def __init__(self, method):
        self.method = method
        self.best_clusterer = None
        self.best_n_clusters = None
        self.best_score = self.init_best_score()


    def _is_higher_better_metric(self):
        return self.method in ('silhouette', 'calinski-harabasz')


    def _is_lower_better_metric(self):
        return self.method == 'davies-bouldin'


    def init_best_score(self):
        if self._is_higher_better_metric():
            return -float("inf")
    
        elif self._is_lower_better_metric():
            return float("inf")
        
        raise Exception("Method not identified")


    def should_update(self, score):
        if self._is_higher_better_metric():
            return score > self.best_score
        
        elif self._is_lower_better_metric():
            return score < self.best_score
        
        raise Exception("Method not identified")


    def evaluate(self, clusterer_model, score, n_clusters):       
        if self.should_update(score):
            self.best_clusterer = clusterer_model
            self.best_n_clusters = n_clusters
            self.best_score = score


def run_clustering(descriptors: list[np.ndarray], clustering_config: ClusteringConfig, cluster_evaluator: ClusterEvaluator):

    method = config.CLUSTER_EVAL_METHOD
    sample_size = config.CLUSTER_EVAL_SAMPLE_SIZE
    n_samples = config.CLUSTER_EVAL_N_SAMPLES

    results = defaultdict(list)

    for n_clusters in clustering_config.range_clusters:

        start = time.time()

        clusterer = MiniBatchKMeans(
            n_clusters         = n_clusters,
            random_state       = 42,
            max_iter           = 100,
            batch_size         = clustering_config.batch_size,
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

        if clustering_config.batch_size < config.BATCH_SIZE:
            logging.info(f"Running Kmeans with {n_clusters} clusters...")
        else:
            logging.info(f"Running Batch Kmeans with {n_clusters} clusters...")
        
        # One row over the other
        if isinstance(descriptors, list):
            descriptors = np.concatenate(descriptors, axis=0)
        clusterer.fit(descriptors)
        
        end = time.time()

        logging.info(f"Finished running KMeans. Took {end - start:.1f} sec.")
        
        cluster_evaluation_scores = calc_sampled_cluster_score(
            clusterer,
            descriptors,
            sample_size,
            n_samples,
            method
        )

        mean_score = np.mean(cluster_evaluation_scores['scores'][method])
        std_score = np.std(cluster_evaluation_scores['scores'][method], ddof=1)

        logging.info(f"Mean '{method}' score for {n_clusters} clusters: {mean_score:.3f} ± {std_score:.3f}")
        logging.info(f"Total inertia of {clusterer.inertia_/1024:.1f} K")

        cluster_evaluator.evaluate(clusterer, mean_score, n_clusters)

        # Save information of this run
        results['method'].append(method)
        results['n_clusters'].append(n_clusters)
        results['score'].append(mean_score)
        results['inertia'].append(clusterer.inertia_)
        results['time'].append(end-start)
        results['silhouette'].append(np.mean(cluster_evaluation_scores['scores']['silhouette']))
        results['davies-bouldin'].append(np.mean(cluster_evaluation_scores['scores']['davies-bouldin']))
        results['calinski-harabasz'].append(np.mean(cluster_evaluation_scores['scores']['calinski-harabasz']))
    
    logging.info(f"Kmeans selected number of clusters: {cluster_evaluator.best_n_clusters}.")
    
    return results


def create_codebook(descriptors: list[np.ndarray]):
    """
    Create a dictionary of visual words (visual vocabulary, codebook) out of 
    the given descriptors. Instead of having an infinite number of possible
    points in the space, reduce the possibilities to a certain fixed number of
    clusters.
    
    How to choose vocabulary size (number of clusters)? 
    - Too small: visual words not representative of all patches
    - Too large: quantization artifacts, overfitting

    Args
        descriptors: 
    """

    #logging.info(f"Number of descriptors in the dataset: {descriptors.shape[0]}")
    
    clustering_config = ClusteringConfig(config, len(descriptors))
    cluster_evaluator = ClusterEvaluator(config.CLUSTER_EVAL_METHOD)
    
    results = run_clustering(descriptors, clustering_config, cluster_evaluator)

    if len(clustering_config.range_clusters) > 1:
        plot_results(results)

    # Coordinates of cluster centers. 
    # n_clusters x N, where N is the descriptor size
    codebook = cluster_evaluator.best_clusterer.cluster_centers_
    print(codebook.shape)
    
    return cluster_evaluator.best_clusterer, codebook


def calc_sampled_cluster_score(clusterer, descriptors: np.ndarray, sample_size: int, repeat: int, eval_method: str):
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

    For calinski-harabasz:
        Higher is better.
        The index is the ratio of the sum of between-clusters dispersion and 
        of inter-cluster dispersion for all clusters 
        (where dispersion is defined as the sum of distances squared).
    """

    dataset_size = descriptors.shape[0]
    sample_size = np.min([dataset_size, sample_size])
    logging.info(f"Calculating mean sampled (n={sample_size}) '{eval_method}' score...")

    scores = {
        'silhouette' : [],
        'davies-bouldin': [],
        'calinski-harabasz': [],
    }
    
    for _ in range(repeat):
        
        sample_idxs = random.sample(range(dataset_size), sample_size)
        descriptors_sample = descriptors[sample_idxs, :]
        cluster_idx_sample = clusterer.predict(descriptors_sample).ravel()
        
        scores['silhouette'].append(silhouette_score(descriptors_sample, cluster_idx_sample))
        scores['davies-bouldin'].append(davies_bouldin_score(descriptors_sample, cluster_idx_sample))
        scores['calinski-harabasz'].append(calinski_harabasz_score(descriptors_sample, cluster_idx_sample))

    return { 'scores': scores }


def bovw(descriptions: list[np.ndarray]):
    """
    Args:
        descriptions: vector of extracted features
    """

    ###########################################################################
    # Vocabulary construction (clustering)
    ###########################################################################

    logging.info(f"Running BOVW")

    # The cluster centroids (codebook) are the dictionary of visual words
    clusterer, codebook = create_codebook(descriptions)
    n_clusters = codebook.shape[0]

    print(f"{n_clusters} clusters")

    print(f"There are {len(descriptions)} descriptions")


    ###########################################################################
    # Image modeling
    ###########################################################################
    # Each image is modelled as a histogram which tracks the frequency of 
    # clusters (i.e. visual word, centroid). 
    # 1) For each feature vector of an image, find the cluster index to which it
    #    belongs.
    # 2) Create a histogram where the frequency of each cluster index is tracked

    clusters_histograms = np.zeros((len(descriptions), n_clusters))

    print(f"len(descriptions): {len(descriptions)}")

    for i, des in enumerate(descriptions):
        clusters_idxs = clusterer.predict(des)
        values,_ = np.histogram(clusters_idxs, bins=n_clusters)
        clusters_histograms[i] = values

    pipeline = Pipeline([
        # Transform a count matrix to a normalized tf or tf-idf representation
        # ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
        ('tfidf', OkapiTransformer()),
    ])

    bovw_histogram = pipeline.fit_transform(clusters_histograms).todense()

    return bovw_histogram, clusterer, codebook, pipeline


def  main():
    """
    Extract image features from all the images found in `config.DATA_FOLDER_PATH`.
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

    if config.DESCRIPTIONS_PATH.exists():
        logging.info("Loading descriptions from local file.")
        descriptions_dict, = joblib.load(str(config.DESCRIPTIONS_PATH))
    
    else:
        logging.info("Recalculating descriptions.")

        # Apparently, SIFT can't get from one processes to another
        if DESCRIPTORS.get('corners') and (DESCRIPTORS.get('corners').kind == 'sift'):
            descriptions_dict = extract_descriptors(images_paths, DESCRIPTORS)
        elif config.MULTIPROCESS:
            descriptions_dict = multiprocessed_descriptors_extraction(images_paths, DESCRIPTORS, n_jobs=config.N_JOBS)
        else:
            descriptions_dict = extract_descriptors(images_paths, DESCRIPTORS)
        
        joblib.dump((descriptions_dict,), str(config.DESCRIPTIONS_PATH), compress=3)


    ###########################################################################
    # Concatenate features:
    # Concatenate all the features obtained for one image
    ###########################################################################

    features = []
    for descriptor_name, descriptions in descriptions_dict.items():

        # Descriptions is a list of arrays of size (n,136)
        logging.info(f"Using descriptor '{descriptor_name}'")

        # I expect this to be a list of arrays of size (n,136)
        print(type(descriptions))
        print(len(descriptions))
        
        if descriptor_name == 'corners':
            bovw_histogram, clusterer, codebook, pipeline = bovw(descriptions)
            logging.info(f"Histogram shape: {bovw_histogram.shape}.")
            features.append(bovw_histogram)
        else:
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
        
        to_save['clusterer'] = clusterer
        to_save['codebook'] = codebook
        to_save['pipeline'] = pipeline

    
    joblib.dump(to_save, str(config.BOVW_PATH), compress=3)
    logging.info("Done")


if __name__ == "__main__":
    main()