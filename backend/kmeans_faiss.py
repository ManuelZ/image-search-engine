import faiss
import numpy as np


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
        """ """

        # Can also use faiss.Clustering
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
