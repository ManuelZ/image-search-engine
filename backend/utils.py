# Built-in imports
import base64
import io
from pathlib import Path

# Standard Library imports
import logging

# External imports
import numpy as np
import cv2
import faiss
from PIL import Image
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _document_frequency
from sklearn.utils.validation import check_array, FLOAT_DTYPES
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

# Local imports
from config import Config

config = Config()
rs = np.random.RandomState(42)


def chunkIt(seq, num):
    """Divide a list into roughly equal parts
    From: https://stackoverflow.com/a/2130035/1253729
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg

    return out


def get_image(image_path):
    """
    Load an image from disk and return a thumbnail of it.
    The image is only used for display purposes.
    """
    config = Config()
    size = config.THUMBNAIL_SIZE, config.THUMBNAIL_SIZE
    try:
        img = Image.open(image_path, mode="r")
    except FileNotFoundError:
        return None
    img.thumbnail(size, Image.LANCZOS)
    img_byte_arr = io.BytesIO()
    try:
        img.save(img_byte_arr, format="JPEG")
    except OSError:
        img.save(img_byte_arr, format="PNG")
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode("ascii")
    return encoded_img


def dhash(image, hashSize=8):
    """
    From: https://pyimagesearch.com/2017/11/27/image-hashing-opencv-python/
    """
    # resize the input image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(image, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])


def convert_hash(h):
    """From: https://pyimagesearch.com/2019/08/26/building-an-image-hashing-search-engine-with-vp-trees-and-opencv/"""
    return int(np.array(h, dtype="float64"))


def hamming(a, b):
    """Compute and return the Hamming distance between the integers
    From: https://pyimagesearch.com/2019/08/26/building-an-image-hashing-search-engine-with-vp-trees-and-opencv/
    """
    return bin(int(a) ^ int(b)).count("1")


def chi2_distance(histA, histB, eps=1e-10):
    """
    From Adrian Rosebrock's Pyimagesearch
    """
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])

    return d


class OkapiTransformer(TransformerMixin, BaseEstimator):
    """
    Modified from:
    https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/feature_extraction/text.py
    According to:
    "Fusion of tf.idf Weighted Bag of Visual Features for Image Classification"

    Check for theory:
    https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html

    """

    def __init__(self, *, norm="l2", use_idf=True, k1=1, k2=1, b=0.75):
        self.norm = norm
        self.use_idf = use_idf
        self.k1 = k1
        self.k2 = k2
        self.b = b

    def fit(self, X, y=None):
        """
        Learn the idf vector (global term weights).

        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.
        """

        X = check_array(X, accept_sparse=("csr", "csc"))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = X.dtype if X.dtype in FLOAT_DTYPES else np.float64

        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            d = {}
            if not sp.issparse(df):
                d["copy"] = False
            df = df.astype(dtype, **d)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))

            self._idf_diag = sp.diags(
                diagonals=idf,
                offsets=0,
                shape=(n_features, n_features),
                format="csr",
                dtype=dtype,
            )

        return self

    def transform(self, X, copy=True):
        """
        Transform a count matrix to a tf or tf-idf representation

        Parameters
        ----------
        X : sparse matrix of (n_samples, n_features)
            a matrix of term/token counts
        copy : bool, default=True
            Whether to copy X and operate on the copy or perform in-place
            operations.
        Returns
        -------
        vectors : sparse matrix of shape (n_samples, n_features)
        """

        X = check_array(X, accept_sparse="csr", dtype=FLOAT_DTYPES, copy=copy)
        if not sp.issparse(X):
            X = sp.csr_matrix(X, dtype=np.float64)

        n_samples, n_features = X.shape

        ######################################################################
        # This part is modified from:
        # https://github.com/arosh/BM25Transformer/blob/master/bm25.py

        # Document length: number of words per document
        dl = X.sum(axis=1)

        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]

        # Number of words used to represent each document.
        # In each row, repeat `dl` for `sz` times
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)

        # Average document length
        avgdl = np.mean(dl)
        ######################################################################

        X.data *= self.k1
        X.data /= X.data + self.k2 * (1 - self.b + self.b * (rep / avgdl))

        return X

    @property
    def idf_(self):
        # if _idf_diag is not set, this will raise an attribute error,
        # which means hasattr(self, "idf_") is False
        return np.ravel(self._idf_diag.sum(axis=0))

    @idf_.setter
    def idf_(self, value):
        value = np.asarray(value, dtype=np.float64)
        n_features = value.shape[0]
        self._idf_diag = sp.spdiags(
            value, diags=0, m=n_features, n=n_features, format="csr"
        )

    def _more_tags(self):
        return {"X_types": "sparse"}


def get_images_paths() -> list[Path]:
    """
    Get all the images paths from `config.DATA_FOLDER_PATH`.
    """
    config = Config()

    images_paths = []
    for ext in config.EXTENSIONS:
        images_paths.extend(config.DATA_FOLDER_PATH.rglob(ext))

    return images_paths


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


def create_search_index(data_array, index_type="cosine"):
    """ """

    num_features = data_array.shape[1]

    # Read this before creating flat indexes for low-dimensional data
    # https://github.com/facebookresearch/faiss/issues/3245
    if index_type == "cosine":
        # https://github.com/facebookresearch/faiss/issues/95#issuecomment-297049159
        index = faiss.IndexFlatIP(num_features)
        faiss.normalize_L2(data_array)

    elif index_type == "l2":
        index = faiss.IndexFlatL2(num_features)
        # faiss.normalize_L2(data_array)

    # With 1K images, there is zero speed improvement,
    # probably because making the inference takes more time than the actual search
    elif index_type == "cell-probe":
        # See the chapter about IndexIVFFlat for the setting of ncentroids.
        ncentroids = 8

        # The code_size, m, is typically a power of two between 4 and 64.
        m = 16

        # Like for IndexPQ, d should be a multiple of m.
        d = num_features

        coarse_quantizer = faiss.IndexFlatL2(d)

        index = faiss.IndexIVFPQ(coarse_quantizer, d, ncentroids, m, 8)
        index.nprobe = 5  # find n most similar clusters
        index.train(data_array)

    index.add(data_array)
    print(f"There are {index.ntotal} images in the search index.")

    return index

