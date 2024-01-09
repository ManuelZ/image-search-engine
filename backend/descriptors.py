# Standard-Library imports
import multiprocessing as mp
from collections import defaultdict
from typing import Protocol

# External imports
import cv2
import numpy as np
from tqdm import tqdm
from skimage import feature
from skimage.util import img_as_ubyte
from imutils import resize
import skimage.transform
import matplotlib.pyplot as plt
import joblib

# Local imports
from utils import dhash, chunkIt
from config import Config


class SupportsDescribe(Protocol):
    def describe(self, image: np.ndarray) -> np.ndarray:
        ...


config = Config()


class Describer:
    def __init__(self, descriptors: dict[str, SupportsDescribe]):
        self.descriptors = self.validate_descriptors(descriptors)

    def validate_descriptors(
        self, descriptors: dict[str, SupportsDescribe]
    ) -> dict[str, SupportsDescribe]:
        """
        Validate that descriptors are provided.
        """
        if len(descriptors.items()) == 0:
            raise Exception("No descriptors provided")

        return descriptors

    def describe(
        self, images_paths, n=1, multiprocess=False
    ) -> dict[str, list[np.ndarray]]:
        """

        Args:
            images_paths:
            n:

        Returns:

        """
        if len(self.descriptors.items()) == 0:
            raise Exception("No descriptors provided")

        extracted: dict[str, list[np.ndarray]] = defaultdict(list)

        if not multiprocess:
            pbar = tqdm(total=len(images_paths))

        for img_path in images_paths.ravel().tolist():
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            image = resize(image, width=config.RESIZE_WIDTH)
            image = img_as_ubyte(image)

            for d_name, descriptor in self.descriptors.items():
                try:
                    # Shape (n, 136), n depends on image size and other factors
                    description = descriptor.describe(image)
                    if description.ndim == 1:
                        print(
                            f"Description has a single dimension ({description.shape}, so it will be reshaped with `.reshape(1,-1)`)"
                        )
                        description = description.reshape(1, -1)
                    # Concatenate all descriptors
                    extracted[d_name].append(description)

                # Brisk sometimes may not find corners
                except Exception as e:
                    print(f"Trouble describing image {img_path}\n {e}")
                    continue

            if not multiprocess:
                pbar.update(1)

        return extracted

    def multiprocessed_describe(self, images_paths, n_jobs=4) -> dict[str, list]:
        """
        Extract images' descriptions using multiple processes.
        """

        pbar = tqdm(total=len(images_paths))

        # https://github.com/tqdm/tqdm#nested-progress-bars
        pool = mp.Pool(processes=n_jobs)
        paths_chunks = chunkIt(images_paths, n_jobs * 16)

        def update_pbar(x):
            num_images = len(list(x.values())[0])
            pbar.update(num_images)

        def error_cb(x):
            print(f"Error while applying async function: ", x)

        results = [
            pool.apply_async(
                func=self.describe,
                args=(paths, i, True),  # multiprocess=True
                callback=update_pbar,
                error_callback=error_cb,
            )
            for i, paths in enumerate(paths_chunks)
        ]

        pool.close()

        extracted = defaultdict(list)
        for r in results:
            # Each .get() call blocks until the applied function is completed
            output = r.get()

            for d_name in self.descriptors.keys():
                extracted[d_name].extend(output[d_name])

        print("All feature extraction processes finished.")
        return extracted


# TODO: currently it's used for extracting corners, make a different function for that and keep this general
def describe_dataset(
    describer: Describer, images_paths: np.ndarray, prediction=False
) -> list[np.ndarray]:
    """ """

    corner_descriptions_path = config.BOVW_CORNER_DESCRIPTIONS_PATH

    if corner_descriptions_path.exists() and not prediction:
        print("Loading dataset features from local file.")
        descriptions_dict = joblib.load(str(corner_descriptions_path))
    else:
        print(f"Extracting features from dataset of {images_paths.shape[0]} images")
        if config.N_JOBS > 1 and not prediction:
            descriptions_dict = describer.multiprocessed_describe(
                images_paths, n_jobs=config.N_JOBS
            )
        else:
            descriptions_dict = describer.describe(images_paths)

        if not prediction:
            joblib.dump(descriptions_dict, str(corner_descriptions_path), compress=3)

    descriptions = descriptions_dict["corners"]

    return descriptions


class CornerDescriptor:
    """
    TODO: explain what is returned
    """

    def __init__(self, name="sift"):
        self.name = name

        if self.name == "brisk":
            self.extractor = cv2.BRISK.create(thresh=30)

        elif self.name == "sift":
            # Creates an array of n_keypoints x nfeatures
            self.extractor = cv2.SIFT.create(nfeatures=128)

    def describe(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: 2 channels gray uint8 image in range 0-255
        Returns:
            an nx64 vector in the case of BRISK (variable number of rows,
            depending on the number of keypoints detected)
        """

        # Convert to grasycale
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = img_as_ubyte(image)

        if self.name in ["brisk", "sift"]:
            kp, des = self.extractor.detectAndCompute(image, mask=None)

        elif self.name == "daisy":
            des = feature.daisy(
                image=image,
                step=32,  # Distance between descriptor sampling points.
                radius=32,  # Radius (in pixels) of the outermost ring.
                rings=2,
                histograms=8,
                orientations=8,
                normalization="daisy",  # 'l1', 'l2', 'daisy', 'off'
            )

            # - The number of daisy descriptors return depends on the size of the image,
            #   the step and radius.
            # - The size of the Daisy vector is: (rings * histograms + 1) * orientations

            des_num = des.shape[0] * des.shape[1]
            des = des.reshape(des_num, des.shape[2])
        else:
            raise Exception("The requested corner descriptor is unrecognized.")

        return des


class HOGDescriptor:
    def __init__(self):
        pass

    def describe(self, image: np.ndarray):
        H = feature.hog(
            image=image,
            orientations=9,
            pixels_per_cell=(32, 32),
            cells_per_block=(2, 2),
            feature_vector=True,
            block_norm="L2-Hys",  # "L1", "L1-sqrt", "L2", "L2-Hys"
        )

        return H


class ColorDescriptor:
    """
    Modified from:
    https://www.pyimagesearch.com/2014/12/01/complete-guide-building-image-search-engine-python-opencv/

    3D color histogram in the HSV color space with:
    - 8 bins for the Hue channel
    - 12 bins for the saturation channel
    - 3 bins for the value channel

    Yielding a total feature vector of dimension 8 x 12 x 3 = 288.
    """

    def __init__(self, bins=(8, 12, 3)):
        self.bins = bins

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist(
            [image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256]
        )

        hist = cv2.normalize(hist, hist).flatten()

        return hist

    def describe(self, image: np.ndarray):
        """
        Return a
        """

        # To HSV color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        (h, w) = image.shape[:2]

        # Compute the center of the image
        (cX, cY) = (int(w * 0.5), int(h * 0.5))

        # Divide the image into four rectangles/segments:
        # top-left, top-right, bottom-right, bottom-left
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        # Construct an elliptical mask representing the center of the image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, [255, 255, 255], -1)

        features = []
        for startX, endX, startY, endY in segments:
            # Construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            # Extract a color histogram from the image, then update the feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)

        # Extract a color histogram from the elliptical region and update the
        # feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)

        return features


class DHashDescriptor:
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    def describe(self, image: np.ndarray):
        return np.array([dhash(image, self.hash_size)]).reshape(1, 1)


class ColorMomentHashDescriptor:
    def describe(self, image):
        return np.array([cv2.img_hash.colorMomentHash(image)]).reshape(1, -1)
