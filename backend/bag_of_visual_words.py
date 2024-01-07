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
from pathlib import Path
from typing import Protocol

# External imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.pipeline import Pipeline
import faiss
import joblib

# Local imports
from utils import OkapiTransformer
from config import Config
from descriptors import Describer, CornerDescriptor


class SupportsFit(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        ...


class SupportsPredict(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


class SKLearnKMeansLike(SupportsFit, SupportsPredict, Protocol):
    cluster_centers_: np.ndarray
    ...


config = Config()


def plot_results(results):
    num_graphs = 5

    ax = plt.subplot(num_graphs, 1, 1)
    plt.plot(results["n_clusters"], results["time"])
    plt.title("Training time [seconds]")

    plt.subplot(num_graphs, 1, 2, sharex=ax)
    plt.plot(results["n_clusters"], results["inertia"])
    plt.title("Inertia (lower is better)")

    plt.subplot(num_graphs, 1, 3, sharex=ax)
    plt.plot(results["n_clusters"], results["davies-bouldin"])
    plt.title(f"Davies-Bouldin (lower is better)")

    plt.subplot(num_graphs, 1, 4, sharex=ax)
    plt.plot(results["n_clusters"], results["silhouette"])
    plt.title(f"Silhouette (higher is better)")

    plt.subplot(num_graphs, 1, 5, sharex=ax)
    plt.plot(results["n_clusters"], results["calinski-harabasz"])
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
                (self.max_clusters - self.min_clusters)
                / self.config.NUM_CLUSTERS_TO_TEST,
                dtype=np.int32,
            )

        logging.info(f"Number of clusters to test: {clusters_range}")

        return clusters_range


class FaissKMeans:
    """
    Modified from:
    https://towardsdatascience.com/k-means-8x-faster-27x-lower-error-than-scikit-learns-in-25-lines-eaedc7a3a0c8

    Docs: https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization#clustering
    """

    def __init__(self, n_clusters=8, n_init=3, max_iter=25):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.inertia_: float
        self.cluster_centers_: np.ndarray | None
        self.kmeans: faiss.Kmeans

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """
        nredo: run the clustering this number of times, and keep the best centroids (selected according to clustering objective)
        """
        self.kmeans = faiss.Kmeans(
            seed=42,
            d=int(X.shape[1]),
            k=int(self.n_clusters),
            niter=self.max_iter,
            nredo=self.n_init,
            # update_index = True,
            spherical=True,
            verbose=False,
        )
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        # I: The nearest centroid for each line vector in x.
        # D: contains the squared L2 distances
        D, I = self.kmeans.index.search(X.astype(np.float32), 1)  # type: ignore
        return I


class ClusteringEvaluator:
    def __init__(self, method):
        self.method = method
        self.best_clusterer: SKLearnKMeansLike
        self.best_n_clusters: int
        self.best_score = self.init_best_score()

    def _is_higher_better_metric(self):
        return self.method in ("silhouette", "calinski-harabasz")

    def _is_lower_better_metric(self):
        return self.method == "davies-bouldin"

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

    def evaluate(self, clusterer_model: SKLearnKMeansLike, score, n_clusters):
        if self.should_update(score):
            self.best_clusterer = clusterer_model
            self.best_n_clusters = n_clusters
            self.best_score = score


def run_clustering(
    descriptions_list: list[np.ndarray],
    clustering_config: ClusteringConfig,
    clustering_evaluator: ClusteringEvaluator,
):
    method = config.CLUSTER_EVAL_METHOD
    sample_size = config.CLUSTER_EVAL_SAMPLE_SIZE
    n_samples = config.CLUSTER_EVAL_N_SAMPLES

    results = defaultdict(list)

    for n_clusters in clustering_config.range_clusters:
        start = time.time()

        # clusterer = MiniBatchKMeans(
        #     n_clusters = n_clusters,
        #     random_state = 42,
        #     max_iter = 100,
        #     batch_size = clustering_config.batch_size,
        #     verbose = 0,
        #     tol = 0, # tune this
        #     max_no_improvement = 5,
        #     n_init = 3
        # )

        clusterer = FaissKMeans(n_clusters)

        # One row over the other
        descriptions = np.concatenate(descriptions_list, axis=0)

        clusterer.fit(descriptions)

        end = time.time()

        logging.info(f"Finished running KMeans. Took {end - start:.1f} sec.")

        cluster_evaluation_scores = calc_sampled_cluster_score(
            clusterer, descriptions, sample_size, n_samples, method
        )

        mean_score = np.mean(cluster_evaluation_scores[method])
        std_score = np.std(cluster_evaluation_scores[method], ddof=1)

        logging.info(
            f"Mean '{method}' score for {n_clusters} clusters: {mean_score:.3f} ± {std_score:.3f}"
        )

        logging.info(f"Total inertia: {clusterer.inertia_/1024:.1f} K")

        clustering_evaluator.evaluate(clusterer, mean_score, n_clusters)

        # Save information of this run
        results["method"].append(method)
        results["n_clusters"].append(n_clusters)
        results["score"].append(mean_score)
        results["inertia"].append(clusterer.inertia_)
        results["time"].append(end - start)
        results["silhouette"].append(np.mean(cluster_evaluation_scores["silhouette"]))
        results["davies-bouldin"].append(
            np.mean(cluster_evaluation_scores["davies-bouldin"])
        )
        results["calinski-harabasz"].append(
            np.mean(cluster_evaluation_scores["calinski-harabasz"])
        )

    logging.info(
        f"Kmeans selected number of clusters: {clustering_evaluator.best_n_clusters}."
    )

    return results


def create_codebook(
    descriptions: list[np.ndarray],
) -> tuple[SKLearnKMeansLike, np.ndarray]:
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


def calc_sampled_cluster_score(
    clusterer, descriptors: np.ndarray, sample_size: int, repeat: int, eval_method: str
):
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
        "silhouette": [],
        "davies-bouldin": [],
        "calinski-harabasz": [],
    }

    for _ in range(repeat):
        sample_idxs = random.sample(range(dataset_size), sample_size)
        descriptors_sample = descriptors[sample_idxs, :]
        cluster_idx_sample = clusterer.predict(descriptors_sample).ravel()

        scores["silhouette"].append(
            silhouette_score(descriptors_sample, cluster_idx_sample)
        )
        scores["davies-bouldin"].append(
            davies_bouldin_score(descriptors_sample, cluster_idx_sample)
        )
        scores["calinski-harabasz"].append(
            calinski_harabasz_score(descriptors_sample, cluster_idx_sample)
        )

    return scores


def extract_bovw_features(
    descriptions: list[np.ndarray], codebook: np.ndarray, clusterer: SKLearnKMeansLike
) -> tuple[np.ndarray, Pipeline]:
    ###########################################################################
    # Image modeling
    ###########################################################################
    # Each image is modelled as a histogram which tracks the frequency of
    # clusters (i.e. visual word, centroid).
    # 1) For each feature vector of an image, find the cluster index to which it
    #    belongs.
    # 2) Create a histogram where the frequency of each cluster index is tracked

    n_clusters = codebook.shape[0]

    clusters_histograms = np.zeros((len(descriptions), n_clusters))

    for i, des in enumerate(descriptions):
        clusters_idxs = clusterer.predict(des)
        values, _ = np.histogram(clusters_idxs, bins=n_clusters)
        clusters_histograms[i] = values

    pipeline = Pipeline(
        [
            # Transform a count matrix to a normalized tf or tf-idf representation
            ("tfidf", OkapiTransformer()),
        ]
    )

    # <class 'scipy.sparse._csr.csr_matrix'>
    bovw_histogram = pipeline.fit_transform(clusters_histograms).toarray()

    logging.info(f"Histogram shape: {bovw_histogram.shape}.")
    # Shape: (n_images, n_clusters)

    return bovw_histogram, pipeline


def train_bag_of_visual_words(
    images_paths: list[Path],
) -> tuple[SKLearnKMeansLike, np.ndarray, list[np.ndarray]]:
    """
    Extract features from
    """

    if config.BOVW_CORNER_DESCRIPTIONS_PATH.exists():
        logging.info("Loading corner features for a BOVW model from local file.")
        (descriptions_dict,) = joblib.load(str(config.BOVW_CORNER_DESCRIPTIONS_PATH))
    else:
        logging.info("Extracting corner features for a BOVW model.")
        # Extract corner features and describe all images
        describer = Describer({"corners": CornerDescriptor("daisy")})

        # TODO: Duplicated. create a function.
        if config.MULTIPROCESS:
            descriptions_dict = describer.multiprocessed_descriptors_extraction(
                images_paths, n_jobs=config.N_JOBS
            )
        else:
            descriptions_dict = describer.generate_descriptions(images_paths)

        joblib.dump(
            (descriptions_dict,), str(config.BOVW_CORNER_DESCRIPTIONS_PATH), compress=3
        )

    descriptions = descriptions_dict["corners"]

    # The cluster centroids (codebook) are the dictionary of visual words
    clusterer, codebook = create_codebook(descriptions)

    return clusterer, codebook, descriptions
