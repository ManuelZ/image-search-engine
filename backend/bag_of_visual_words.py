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
from sklearn.base import BaseEstimator
import pandas as pd

# Local imports
from utils import OkapiTransformer
from utils import chunkIt
from config import Config
from descriptors import Describer, describe_dataset
from kmeans_faiss import FaissKMeans

rs = np.random.RandomState(42)


config = Config()


def run_clustering(
    descriptions_list: list[np.ndarray],
    n_clusters: int,
):
    clusterer = FaissKMeans(n_clusters)
    descriptions = np.concatenate(descriptions_list, axis=0)
    clusterer.fit(descriptions)
    return clusterer


class BOVW(BaseEstimator):
    """
    Bag of Visual Words modelling.

    1) Each image is described by multiple feature vectors (corner descriptions).

    2) Train a KMeans model to create a dictionary of visual words (visual vocabulary, codebook)
       out of the given images' descriptions. Clustering aims to limit the infinite potential
       points in space to a fixed number of clusters.

    3) For each feature vector of an image, find the cluster index to which it belongs.

    4) Create a histogram where the frequency of each cluster index is tracked.

    The histogram is the final feature vector with which an image is described.

    Note: The codebook, i.e. the cluster centers, are stored inside the clusterer object.
    """

    def __init__(self, describer, n_clusters=10):
        self.describer: Describer = describer
        self.n_clusters: int = n_clusters
        self.clusterer: FaissKMeans

    def fit(self, X: np.ndarray, y=None):
        """
        Tran a KMeans model. Perform steps 1 and 2 of the explanation above.

        Args:
            X: file paths
        """
        self.descriptions = describe_dataset(self.describer, X)
        self.clusterer = run_clustering(self.descriptions, self.n_clusters)
        return self

    def transform(self, X: np.ndarray, y=None):
        """
        Perform steps 3 and 4 of the explanation above.

        Args:
            X: file paths

        """

        def create_visual_word_histogram(images_descriptions: list[np.ndarray]):
            """ """
            clusters_histograms = np.zeros((len(images_descriptions), self.n_clusters))
            for i, X in enumerate(images_descriptions):
                # Quantization
                clusters_indexes = self.clusterer.transform(X)
                values, _ = np.histogram(clusters_indexes, bins=self.n_clusters)
                clusters_histograms[i] = values

            return clusters_histograms

        # TODO: Clean this up. This is a hacky way of doing things.
        if self.descriptions is None:
            descriptions = describe_dataset(self.describer, X, prediction=True)
        else:
            descriptions = self.descriptions

        if config.MULTIPROCESS:
            descriptions_chunks = chunkIt(descriptions, config.N_JOBS)
            with parallel_config(backend="threading", n_jobs=config.N_JOBS):
                list_of_histograms = Parallel()(
                    delayed(create_visual_word_histogram)(des)
                    for des in descriptions_chunks
                )
            clusters_histograms = np.concatenate(list_of_histograms)
        else:
            clusters_histograms = create_visual_word_histogram(descriptions)

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
    """ """

    X = np.array([image_path])
    bovw_histogram = pipeline.transform(X).todense()
    return bovw_histogram


def generate_bovw_features_from_descriptions(
    images_paths: np.ndarray, describer: Describer
):
    print(f"Received {images_paths.shape[0]} images to process")

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

    search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "bovw__n_clusters": clusters_to_test,  # e.g. [1, 10, 1000]
        },
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
    # TODO: use a VPTree or a better faiss index
    index = faiss.IndexFlatL2(bovw_histograms.shape[1])
    index.add(bovw_histograms)
    print(f"There are {index.ntotal} images in the index.")

    save_indexes_and_pipeline(
        best_pipeline.named_steps["bovw"].clusterer.index,
        index,
        best_pipeline,
    )


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
