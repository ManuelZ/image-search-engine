# Standard-Library imports
from collections import defaultdict
from typing import Protocol

# External imports
import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from skimage import feature
import joblib
from joblib import Parallel, delayed
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor, BitModel
from tqdm import tqdm

# Local imports
from utils import dhash, chunkIt
from config import Config, DnnModels


class SupportsDescribe(Protocol):
    def describe(self, image: np.ndarray) -> np.ndarray: ...


config = Config()


class CornerDescriptorFactory:
    def get_descriptor(self, name) -> cv2.BRISK | cv2.SIFT | cv2.ORB:
        if name == "brisk":
            return cv2.BRISK.create(thresh=30)
        elif name == "sift":
            return cv2.SIFT.create(nfeatures=128)
        # elif name == "surf":
        #    return cv2.xfeatures2d.SURF.create()
        elif name == "orb":
            return cv2.ORB.create(nfeatures=1024)
        # elif name == "freak":
        #     return cv2.xfeatures2d.FREAK.create()
        else:
            raise Exception("Invalid descriptor name")


class Describer:
    """
    To be able to extract features using multiple descriptors.
    """

    def __init__(self, descriptors: dict[str, SupportsDescribe]):
        self.descriptors = self._validate_descriptors(descriptors)

    def _validate_descriptors(
        self, descriptors: dict[str, SupportsDescribe]
    ) -> dict[str, SupportsDescribe]:
        """Validate that descriptors are provided."""
        if not descriptors:
            raise Exception("No descriptors provided")

        return descriptors

    def read_image(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise Exception("Problem opening image")
        return image.astype(np.uint8)

    def describe(self, images_paths, multiprocess=False) -> dict[str, np.ndarray]:
        """ """

        total_images = len(images_paths)

        if not multiprocess:
            pbar = tqdm(total=total_images)

        descriptions: dict[str, list[np.ndarray]] = defaultdict(list)
        for img_path in images_paths.ravel().tolist():

            try:
                image = self.read_image(img_path)
                for d_name, descriptor in self.descriptors.items():
                    description = descriptor.describe(image)

                    if description is None:
                        raise Exception(f"Couldn't describe image '{img_path.name}'.")

                    if description.ndim == 1:
                        description = description.reshape(1, -1)

                    descriptions[d_name].append(description)

            except Exception as e:
                print(f"ERROR: Problem describing image '{img_path}'\n '{e}'")
                continue

            if not multiprocess:
                pbar.update(1)

        return descriptions


def describe_dataset(
    describer: Describer, images_paths: np.ndarray, prediction=False
) -> list[np.ndarray | torch.Tensor]:
    """ """

    n_jobs = 1 if prediction else config.N_JOBS

    # Load corner descriptions if they exist
    if config.BOVW_CORNER_DESCRIPTIONS_PATH.exists() and not prediction:
        print("Loading corner description features from local file.")
        descriptions = joblib.load(str(config.BOVW_CORNER_DESCRIPTIONS_PATH))

    else:  # Extract new features
        paths_chunks = chunkIt(images_paths, n_jobs * 2)

        print(
            "Extracting features from {} images. Splitting in {} jobs.".format(
                images_paths.shape[0], len(paths_chunks)
            )
        )

        with Parallel(backend="threading", n_jobs=n_jobs) as parallel:
            dicts = parallel(
                delayed(describer.describe)(paths, n_jobs > 1)
                for paths in tqdm(paths_chunks)
            )

        print("Concatenating the results into a single array...")
        descriptions = []
        for descriptions_dict in dicts:
            for key in descriptions_dict.keys():
                for image_description in descriptions_dict[key]:
                    descriptions.append(image_description)
        print("Results concatenated.")

    return descriptions


class CNNDescriptor:
    def __init__(self, model):
        self.model = model
        self.preprocessor = None
        self.feature_extractor = None
        self.initialize_model()

    def initialize_model(self):
        """ """

        if self.model == DnnModels.RESNET:
            self.preprocessor = A.Compose(
                [
                    A.Resize(config.RESIZE_SIZE, config.RESIZE_SIZE, cv2.INTER_LINEAR),
                    A.Normalize(),
                    ToTensorV2(),  # converts to PyTorch CHW format
                ]
            )

            model = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            )

            # Check layers names with `dict(model.named_modules())`
            self.feature_extractor = create_feature_extractor(
                model, return_nodes={"flatten": "features"}
            ).to(config.DEVICE)

        elif self.model == DnnModels.BiT:
            self.preprocessor = AutoImageProcessor.from_pretrained("google/bit-50")
            self.feature_extractor = BitModel.from_pretrained("google/bit-50")

        else:
            raise ValueError(
                f"Model '{self.model}' not recognized for feature extraction"
            )

        self.feature_extractor.eval()

    def extract_features(self, image):
        """ """

        if self.model == DnnModels.RESNET:
            image = self.preprocessor(image=image)["image"].to(config.DEVICE)
            image = image.unsqueeze(0)  # Add batch dimension
            features = self.feature_extractor(image)["features"]

        elif self.model == config.DnnModels.BiT:
            image = self.preprocessor(image, return_tensors="pt")
            features = self.feature_extractor(**image).last_hidden_state

        else:
            raise Exception("Model not recognized")

        return features.cpu().flatten()

    def describe(self, image: np.ndarray):
        """ """

        with torch.no_grad():
            features = self.extract_features(image)

        return features


class CornerDescriptor:
    """
    TODO: explain what is returned
    """

    def __init__(self, name="sift"):
        self.name = name
        self.factory = CornerDescriptorFactory()

    def describe(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: 2 channels gray uint8 image in range 0-255
        Returns:
            an nx64 vector in the case of BRISK (variable number of rows,
            depending on the number of keypoints detected)
        """

        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.astype(np.uint8)

        if self.name in ["brisk", "sift", "surf", "orb", "freak"]:
            # I don't assign extractor to an attribute because Sift can't be seriallized,
            # and that happens when doing Cross Validation with Scikit-Learn
            extractor = self.factory.get_descriptor(self.name)
            if self.name == "freak":
                kp, des = extractor.compute(image, mask=None)
            else:
                kp, des = extractor.detectAndCompute(image, mask=None)

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
