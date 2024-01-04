# Built-in imports
import base64
import io

# External imports
import numpy as np
import cv2
from PIL import Image
import faiss # Windows: conda install faiss-cpu
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _document_frequency
from sklearn.utils.validation import check_array, FLOAT_DTYPES
import scipy.sparse as sp
#import faiss

# Local imports
from config import Config


def chunkIt(seq, num):
    """ Divide a list into roughly equal parts
    From: https://stackoverflow.com/a/2130035/1253729
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
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
        img = Image.open(image_path, mode='r')
    except FileNotFoundError:
        return None
    img.thumbnail(size, Image.LANCZOS)
    img_byte_arr = io.BytesIO()
    try:
        img.save(img_byte_arr, format='JPEG')
    except OSError:
        img.save(img_byte_arr, format='PNG')
    encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
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
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def chi2_distance(histA, histB, eps=1e-10):
    """
    From Adrian Rosebrock's Pyimagesearch
    """
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
        for (a, b) in zip(histA, histB)])

    return d

class FaissKMeans:
    """
    Modified from:
    https://towardsdatascience.com/k-means-8x-faster-27x-lower-error-than-scikit-learns-in-25-lines-eaedc7a3a0c8

    Docs: https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization#clustering
    """
    def __init__(self, n_clusters=8, n_init=3, max_iter=25):
        self.n_clusters       = n_clusters
        self.n_init           = n_init
        self.max_iter         = max_iter
        self.kmeans           = None
        self.inertia_         = None
        self.cluster_centers_ = None

    def fit(self, X, y=None):
        self.kmeans = faiss.Kmeans(
            seed    = 42,
            d       = X.shape[1],
            k       = self.n_clusters,
            niter   = self.max_iter,
            nredo   = self.n_init,
            # update_index = True,
            spherical = True,
            verbose = True
        )
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]


class OkapiTransformer(TransformerMixin, BaseEstimator):
    """
    Modified from:
    https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/feature_extraction/text.py
    According to:
    "Fusion of tf.idf Weighted Bag of Visual Features for Image Classification"

    Check for theory:
    https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html

    """

    def __init__(self, *, norm='l2', use_idf=True, k1=1, k2=1, b=0.75):
        self.norm    = norm
        self.use_idf = use_idf
        self.k1      = k1
        self.k2      = k2
        self.b       = b

    def fit(self, X, y=None):
        """
        Learn the idf vector (global term weights).
        
        Parameters
        ----------
        X : sparse matrix of shape n_samples, n_features)
            A matrix of term/token counts.
        """

        X = check_array(X, accept_sparse=('csr', 'csc'))
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
            idf = np.log( (n_samples - df + 0.5) / (df + 0.5))

            self._idf_diag = sp.diags(
                diagonals = idf, 
                offsets   = 0,
                shape     = (n_features, n_features),
                format    = 'csr',
                dtype     = dtype
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

        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES, copy=copy)
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
        X.data /= (X.data + self.k2 * (1 - self.b + self.b * (rep / avgdl)))

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
            value,
            diags  = 0,
            m      = n_features,
            n      = n_features,
            format = 'csr'
        )

    def _more_tags(self):
        return {'X_types': 'sparse'}
