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
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.pipeline import Pipeline
import faiss

# Local imports
from utils import OkapiTransformer
from config import Config

config = Config()


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
        self.batch_size = self.__calc_batch_size(n_descriptors)
        self.min_clusters = self.config.MIN_NUM_CLUSTERS
        self.max_clusters = np.min([self.config.MAX_NUM_CLUSTERS + 1, n_descriptors])
        self.range_clusters = self.__calc_clusters_range()


    def __calc_batch_size(self, n_descriptors):
        return np.min([n_descriptors, self.config.BATCH_SIZE])


    def __calc_clusters_range(self):
        if self.config.NUM_CLUSTERS_TO_TEST == 1:
            clusters_range = [self.config.MIN_NUM_CLUSTERS]

        else:
            # Split the available range in n clusters
            clusters_range = np.arange(
                self.config.MIN_NUM_CLUSTERS,
                self.max_clusters,
                (self.max_clusters - self.min_clusters) / self.config.NUM_CLUSTERS_TO_TEST,
                dtype=np.int32
            )
        
        logging.info(f"Number of clusters to test: {clusters_range}")
        
        return clusters_range




class ClustererWrapper:
    def __init__(self, batch_size, n_clusters, n_features):
        self.inertia_ = None
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.n_features = n_features  # Number of features of the observations
        self.cluster_centers_ = None

    def set_clusterer(self, clusterer, **kwargs):

        if clusterer == "mini-batch-kmeans":
            self.clusterer = MiniBatchKMeans(
                n_clusters = self.n_clusters,
                random_state = 42,
                max_iter = 100,
                batch_size = self.batch_size,
                verbose = 0,
                tol = 0, # tune this
                max_no_improvement = 5,
                n_init = 3
            )
        
        elif clusterer == "faiss-kmeans":
            # The recipe: play with parameters of k-means clustering niter and max_points_per_centroid.
            # From: https://github.com/facebookresearch/faiss/wiki/How-to-make-Faiss-run-faster#k-means-clustering
            self.clusterer = faiss.Kmeans(int(self.n_features), int(self.n_clusters), **kwargs)
    
    def fit(self, X: np.ndarray):
       
        # faiss KMeans
        if isinstance(self.clusterer, faiss.Kmeans):
            self.clusterer.train(X)
            self.inertia_ = self.clusterer.obj
            self.cluster_centers_ = self.clusterer.centroids
        
        # sklearn Kmeans
        elif isinstance(self.clusterer, MiniBatchKMeans):
            self.clusterer.fit(X)

    def predict(self, X):

        if isinstance(self.clusterer, faiss.Kmeans):
            """
            query n vectors of dimension d to the index.

            return at most k vectors. If there are not enough results for a query, the result array is padded with -1s.

            Parameters:
                n: number of vectors
                x: input vectors to search, size n * d
                k: number of extracted vectors
                distances: output pairwise distances, size n*k
                labels: output labels of the NNs, size n*k
            """
            # I: The nearest centroid for each line vector in x. 
            # D: contains the squared L2 distances
            D, I = self.clusterer.index.search(X, 1)
            return I
        
        elif isinstance(self.clusterer, MiniBatchKMeans):
            return self.clusterer.predict(X)
    

class ClusteringEvaluator:

    def __init__(self, method):
        self.method = method
        self.best_clusterer: ClustererWrapper
        self.best_n_clusters: int
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

    def evaluate(self, clusterer_model: ClustererWrapper, score, n_clusters):       
        if self.should_update(score):
            self.best_clusterer = clusterer_model
            self.best_n_clusters = n_clusters
            self.best_score = score


def run_clustering(
    descriptions: list[np.ndarray],
    clustering_config: ClusteringConfig,
    clustering_evaluator: ClusteringEvaluator
    ):

    method = config.CLUSTER_EVAL_METHOD
    sample_size = config.CLUSTER_EVAL_SAMPLE_SIZE
    n_samples = config.CLUSTER_EVAL_N_SAMPLES

    results = defaultdict(list)

    for n_clusters in clustering_config.range_clusters:

        start = time.time()

        if len(descriptions[0].shape) == 1:
            num_features = descriptions[0].shape[0]
        else:
            num_features = descriptions[0].shape[1]

        clusterer = ClustererWrapper(
            batch_size = clustering_config.batch_size,
            n_clusters = n_clusters,
            n_features= num_features
        )

        clusterer.set_clusterer("faiss-kmeans", niter=25, verbose=True)

        if clustering_config.batch_size < config.BATCH_SIZE:
            logging.info(f"Running Kmeans with {n_clusters} clusters...")
        else:
            logging.info(f"Running Batch Kmeans with {n_clusters} clusters...")
        
        # One row over the other
        if isinstance(descriptions, list):
            descriptions = np.concatenate(descriptions, axis=0)
        
        clusterer.fit(descriptions)
        
        end = time.time()

        logging.info(f"Finished running KMeans. Took {end - start:.1f} sec.")
        
        cluster_evaluation_scores = calc_sampled_cluster_score(
            clusterer,
            descriptions,
            sample_size,
            n_samples,
            method
        )

        mean_score = np.mean(cluster_evaluation_scores[method])
        std_score = np.std(cluster_evaluation_scores[method], ddof=1)

        logging.info(f"Mean '{method}' score for {n_clusters} clusters: {mean_score:.3f} ± {std_score:.3f}")
        # Don't know yet how to get inertia from faiss
        #logging.info(f"Total inertia of {clusterer.inertia_/1024:.1f} K")

        clustering_evaluator.evaluate(clusterer, mean_score, n_clusters)

        # Save information of this run
        results['method'].append(method)
        results['n_clusters'].append(n_clusters)
        results['score'].append(mean_score)
        results['inertia'].append(clusterer.inertia_)
        results['time'].append(end-start)
        results['silhouette'].append(np.mean(cluster_evaluation_scores['silhouette']))
        results['davies-bouldin'].append(np.mean(cluster_evaluation_scores['davies-bouldin']))
        results['calinski-harabasz'].append(np.mean(cluster_evaluation_scores['calinski-harabasz']))
    
    logging.info(f"Kmeans selected number of clusters: {clustering_evaluator.best_n_clusters}.")
    
    return results


def create_codebook(descriptions: list[np.ndarray]) -> tuple[ClustererWrapper, np.ndarray]:
    """
    Create a dictionary of visual words (visual vocabulary, codebook) out of 
    the given descriptions. Instead of having an infinite number of possible
    points in the space, reduce the possibilities to a certain fixed number of
    clusters.
    
    How to choose vocabulary size (number of clusters)? 
    - Too small: visual words not representative of all patches
    - Too large: quantization artifacts, overfitting

    Args:
        descriptions:
    """
    
    clustering_config = ClusteringConfig(config, len(descriptions))
    clustering_evaluator = ClusteringEvaluator(config.CLUSTER_EVAL_METHOD)
    
    results = run_clustering(descriptions, clustering_config, clustering_evaluator)

    if len(clustering_config.range_clusters) > 1:
        plot_results(results)

    best_clusterer = clustering_evaluator.best_clusterer

    # The Codebook has the coordinates of the clusters' centers
    # Shape: n_clusters x N, where N is the descriptor size
    codebook = best_clusterer.cluster_centers_ 
    
    return best_clusterer, codebook


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

    return scores


def create_bovw(
    descriptions: list[np.ndarray]
    ) -> tuple[np.ndarray, ClustererWrapper, np.ndarray, Pipeline]:
    """
    Args:
        descriptions
    """

    ###########################################################################
    # Clustering (vocabulary construction)
    ###########################################################################

    logging.info(f"Training Bag of Visual Words")

    # The cluster centroids (codebook) are the dictionary of visual words
    clusterer, codebook = create_codebook(descriptions)
    n_clusters = codebook.shape[0]


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
        # Transform a count matrix to a normalized tf or tf-idf representation
        ('tfidf', OkapiTransformer()),
    ])

    # <class 'scipy.sparse._csr.csr_matrix'>
    bovw_histogram = pipeline.fit_transform(clusters_histograms).toarray()

    return bovw_histogram, clusterer, codebook, pipeline
