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
import logging
from pathlib import Path
from typing import Protocol
from tempfile import mkdtemp

# External imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.pipeline import Pipeline
import faiss
import joblib
from joblib import Parallel, delayed
from joblib import parallel_config
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator
import pandas as pd

# Local imports
from utils import OkapiTransformer
from utils import chunkIt
from config import Config
from descriptors import Describer, CornerDescriptor, describe_dataset

rs = np.random.RandomState(42)


class SupportsFit(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        ...


class SupportsTransform(Protocol):
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...


class SKLearnKMeansLike(SupportsFit, SupportsTransform, Protocol):
    cluster_centers_: np.ndarray
    n_clusters: int
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

    def __init__(
        self, n_clusters=8, n_init=3, max_iter=25, init_centroids=None, index=None
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.inertia_: float
        self.cluster_centers_: np.ndarray | None
        self.kmeans: faiss.Kmeans
        self.init_centroids = init_centroids
        self.index = index

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """
        nredo: run the clustering this number of times, and keep the best centroids (selected according to clustering objective)
        """

        # Also can use faiss.Clustering
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
        # init_centroids is only passed when KMeans was loaded from a file
        # i.e. when the centroids, also named Codebook, were loaded from a file
        self.kmeans.train(X.astype(np.float32), init_centroids=self.init_centroids)
        self.index = self.kmeans.index
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """ """
        # I: The nearest centroid for each line vector in x.
        L2_distances, I = self.index.search(X.astype(np.float32), 1)  # type: ignore
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
    n_clusters: int,
):
    clusterer = FaissKMeans(n_clusters)
    descriptions = np.concatenate(descriptions_list, axis=0)
    clusterer.fit(descriptions)
    return clusterer


class BOVW(BaseEstimator):
    def __init__(self, describer, n_clusters=10):
        self.describer: Describer = describer
        self.n_clusters: int = n_clusters
        self.clusterer: FaissKMeans

    def fit(self, X: np.ndarray, y=None):
        """
        X: file paths
        """
        self.descriptions = describe_dataset(self.describer, X)
        self.clusterer = run_clustering(self.descriptions, self.n_clusters)
        return self

    def transform(self, X: np.ndarray, y=None):
        def clusterize_and_quantize(descriptions: list[np.ndarray]):
            """
            Create a dictionary of visual words (visual vocabulary, codebook) out of
            the given descriptions. Instead of having an infinite number of possible
            points in the space, reduce the possibilities to a certain fixed number of
            clusters.

            How to choose vocabulary size (number of clusters)?
            - Too small: visual words not representative of all patches
            - Too large: quantization artifacts, overfitting

            """

            clusters_histograms = np.zeros((len(descriptions), self.n_clusters))
            for i, X in enumerate(descriptions):
                clusters_indexes = self.clusterer.transform(X)
                values, _ = np.histogram(clusters_indexes, bins=self.n_clusters)
                clusters_histograms[i] = values

            return clusters_histograms

        if self.descriptions is None:
            descriptions = describe_dataset(self.describer, X, prediction=True)
        else:
            descriptions = self.descriptions

        if config.MULTIPROCESS:
            descriptions_chunks = chunkIt(descriptions, config.N_JOBS)
            with parallel_config(backend="threading", n_jobs=config.N_JOBS):
                list_of_histograms = Parallel()(
                    delayed(clusterize_and_quantize)(des) for des in descriptions_chunks
                )
            clusters_histograms = np.concatenate(list_of_histograms)
        else:
            clusters_histograms = clusterize_and_quantize(descriptions)

        return clusters_histograms

    def fit_transform(self, X: np.ndarray, y=None):
        self.fit(X)
        return self.transform(X)


def calc_sampled_cluster_score(
    estimator,
    X,
    y=None,
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
    scoring_func = davies_bouldin_score
    # sign = 1 if greater_is_better else -1
    # for davies_bouldin_score, the minimum score is zero,
    # with lower values indicating better clustering.
    sign = -1

    bovw = estimator.named_steps["bovw"]
    all_descriptions = np.concatenate(bovw.descriptions)

    # Assign each description to a cluster
    kmeans = bovw.clusterer
    labels_ = kmeans.transform(all_descriptions).ravel()

    dataset_size = all_descriptions.shape[0]
    sample_size = np.min([dataset_size, config.CLUSTER_EVAL_SAMPLE_SIZE])
    logging.info(
        f"Calculating mean sampled (n={sample_size}) 'davies_bouldin_score' score..."
    )

    scores = []
    for _ in range(config.CLUSTER_EVAL_N_SAMPLES):
        sample_idxs = rs.choice(dataset_size, size=sample_size, replace=False)
        X_sample = all_descriptions[sample_idxs]
        labels_sample = labels_[sample_idxs]
        scores.append(scoring_func(X_sample, labels_sample))

    return sign * np.mean(scores)


def generate_bovw_feature(image_path: Path, pipeline: Pipeline):
    """
    Quantize the image descriptor.
    Predict the closest cluster that each sample belongs to. Each value
    returned by predict represents the index of the closest cluster
    center in the code book.
    """
    # clusters_idxs = clusterer.transform(description)
    # n_clusters = clusterer.n_clusters

    # Histogram of image descriptor values
    # query_im_histogram, _ = np.histogram(clusters_idxs, bins=n_clusters)
    # query_im_histogram = query_im_histogram.reshape(1, -1)
    X = np.array([image_path])
    bovw_histogram = pipeline.transform(X).todense()
    return bovw_histogram


def generate_bovw_features_from_descriptions(
    images_paths: np.ndarray,
):
    print(f"Received {images_paths.shape[0]} images to process")

    ###########################################################################
    # Image modeling
    ###########################################################################
    # Each image is modelled as a histogram which tracks the frequency of
    # clusters (i.e. visual word, centroid).
    # 1) For each feature vector of an image, find the cluster index to which it
    #    belongs.
    # 2) Create a histogram where the frequency of each cluster index is tracked
    describer = Describer({"corners": CornerDescriptor(config.CORNER_DESCRIPTOR)})
    pipeline = Pipeline(
        [
            ("bovw", BOVW(describer)),
            ("tfidf", OkapiTransformer()),
        ],
    )

    clusters_to_test = np.unique(
        np.linspace(
            config.MIN_NUM_CLUSTERS,
            config.MAX_NUM_CLUSTERS,
            config.NUM_CLUSTERS_TO_TEST,
        )
        .round()
        .astype(int)
    )

    parameter_grid = {
        "bovw__n_clusters": clusters_to_test,  # e.g. [1, 10, 1000]
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        n_jobs=config.N_JOBS,
        verbose=1,
        scoring=calc_sampled_cluster_score,
    )

    search.fit(images_paths)
    results_df = pd.DataFrame(search.cv_results_)

    print(f"Search finished.")
    print(f"Best score: {search.best_score_:.3f}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Detailed results:")
    print(results_df)

    best_pipeline = search.best_estimator_

    bovw_histograms = best_pipeline.transform(images_paths).todense()

    print(f"Preparing search index...")
    index = faiss.IndexFlatL2(bovw_histograms.shape[1])
    index.add(bovw_histograms)
    print(f"There are {index.ntotal} images in the index.")

    save_indexes_and_pipeline(
        best_pipeline.named_steps["bovw"].clusterer.index,
        index,
        best_pipeline,
    )


def extract_features(image_path: Path, pipeline: Pipeline):
    """ """
    bovw_histogram = generate_bovw_feature(image_path, pipeline)
    return bovw_histogram


def load_cluster_model(
    n_clusters: int,
    index: None | str | Path | faiss.IndexFlatIP | faiss.IndexFlatL2 = None,
):
    """ """
    if isinstance(index, (str, Path)):
        index = faiss.read_index(str(index))

    clusterer = FaissKMeans(n_clusters=n_clusters, index=index)
    return clusterer


def save_indexes_and_pipeline(kmeans_index, index, pipeline: Pipeline):
    """"""
    print("Saving KMeans index", kmeans_index)
    faiss.write_index(kmeans_index, str(config.BOVW_KMEANS_INDEX_PATH))

    print("Saving final index", index)
    faiss.write_index(index, str(config.BOVW_INDEX_PATH))

    print("Saving pipeline", pipeline)
    # Delete faiss kmeans because it can't be pickled
    pipeline.named_steps["bovw"].clusterer = None
    joblib.dump(pipeline, str(config.BOVW_PIPELINE_PATH), compress=3)
