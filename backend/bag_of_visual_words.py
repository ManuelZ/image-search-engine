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

# External imports
import numpy as np
from sklearn.pipeline import Pipeline
import faiss
import joblib
from joblib import Parallel, delayed
from joblib import parallel_config
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import pandas as pd
from tqdm import tqdm

# Local imports
from utils import OkapiTransformer, calc_sampled_cluster_score, create_search_index
from utils import chunkIt
from config import Config
from descriptors import Describer, describe_dataset
from kmeans_faiss import FaissKMeans


config = Config()


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
        self.descriptions: list[np.ndarray]

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

        ################################################################################################################
        # Descriptions
        ################################################################################################################

        if self.descriptions is None:
            descriptions = describe_dataset(self.describer, X, prediction=True)
        else:
            descriptions = self.descriptions

        ################################################################################################################
        # Visual Words
        ################################################################################################################

        def create_visual_word_histogram(images_descriptions: list[np.ndarray]):
            """ """
            clusters_histograms = np.zeros((len(images_descriptions), self.n_clusters))
            for i, X in enumerate(images_descriptions):
                clusters_indexes = self.clusterer.transform(X)  # Quantization
                values, _ = np.histogram(clusters_indexes, bins=self.n_clusters)
                clusters_histograms[i] = values

            return clusters_histograms

        descriptions_chunks = chunkIt(descriptions, config.N_JOBS)
        with parallel_config(backend="threading", n_jobs=config.N_JOBS):
            clusters_histograms = Parallel()(
                delayed(create_visual_word_histogram)(des)
                for des in tqdm(descriptions_chunks)
            )

        clusters_histograms = np.concatenate(clusters_histograms)
        return clusters_histograms

    def fit_transform(self, X: np.ndarray, y=None):
        self.fit(X)
        return self.transform(X)


def run_clustering(
    descriptions: list[np.ndarray],
    n_clusters: int,
):
    """ """

    print(f"Starting clustering...")
    descriptions_arr = np.concatenate(descriptions, axis=0)
    clusterer = FaissKMeans(n_clusters)
    clusterer.fit(descriptions_arr)
    print(f"Clustering finished.")
    return clusterer


def train_bovw_model(images_paths: np.ndarray, describer: Describer):
    """ """

    print(f"Received {images_paths.shape[0]} images to process")

    pipeline = Pipeline(
        [
            ("bovw", BOVW(describer, n_clusters=config.NUM_CLUSTERS)),
            ("tfidf", OkapiTransformer()),
        ],
    )

    if config.BOVW_HYPERPARAMETERS_SEARCH:

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

        pipeline = search.best_estimator_
        bovw_histograms = pipeline.transform(images_paths).todense()
    else:
        bovw_histograms = pipeline.fit_transform(images_paths).todense()

    clusterer = pipeline.named_steps["bovw"].clusterer
    print("Saving KMeans index", clusterer.index)
    faiss.write_index(clusterer.index, str(config.BOVW_KMEANS_INDEX_PATH))

    # NOTE: Faiss works only with float32. There could be floating point precision issues!
    bovw_histograms = bovw_histograms.astype(np.float32)
    index = create_search_index(bovw_histograms)

    print("Saving final index", index)
    faiss.write_index(index, str(config.BOVW_INDEX_PATH))

    print("Saving pipeline", pipeline)

    # Delete faiss kmeans because it can't be pickled
    pipeline.named_steps["bovw"].clusterer = None

    # Delete the corner descriptions because they are not needed for prediction
    pipeline.named_steps["bovw"].descriptions = None

    joblib.dump(pipeline, str(config.BOVW_PIPELINE_PATH), compress=0)


def load_cluster_model(
    n_clusters: int,
    index: None | str | Path | faiss.IndexFlatIP | faiss.IndexFlatL2 = None,
):
    """ """
    if isinstance(index, (str, Path)):
        index = faiss.read_index(str(index))

    clusterer = FaissKMeans(n_clusters=n_clusters, index=index)
    return clusterer
