
# External imports
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from skimage import feature
from skimage import io
from skimage.color import rgb2gray


class CornerDescriptor:
    """
    TODO: explain what is returned
    """

    def __init__(self, kind="brisk"):
        self.kind = kind

        if kind == "brisk":
            self.extractor = cv2.BRISK_create(thresh=30)
        
        elif kind == "sift":
            self.extractor = cv2.SIFT_create()

        
    def describe(self, image):
        """
        Args:
            image: 2 channels gray uint8 image in range 0-255
        Returns:
            an nx64 vector in the case of BRISK (variable number of rows,
            depending on the number of keypoints detected)
        """

        if self.kind in ['brisk',  'sift']:
            #if image.dtype == "float64":
                #image = (image * 255).astype(np.uint8)
            kp, des = self.extractor.detectAndCompute(image, None)
        
        elif self.kind == "daisy":
            des = feature.daisy(
                image        = image,
                step         = 10,  # Distance between descriptor sampling points.
                radius       = 40,  # Radius (in pixels) of the outermost ring.
                rings        = 2,
                histograms   = 6,
                orientations = 8,
                normalization= 'l1', # 'l1', 'l2', 'daisy', 'off'
                visualize    = False # Put to True and uncomment below the plotting code
            )

            # - The number of daisy descriptors return depends on the size of the image,
            # the step and radius.
            # - The size of the Daisy vector is: (rings * histograms + 1) * orientations

            des_num = des.shape[0] * des.shape[1] 
            des = des.reshape(des_num, des.shape[2])

        return des


class HOGDescriptor:
    def __init__(self):
        pass

    def describe(self, image):

        H = feature.hog(
            image           = image,
            orientations    = 9,
            pixels_per_cell = (10, 10),
            cells_per_block = (3, 3), 
            feature_vector  = True,
            block_norm      = "L1-sqrt" # "L1", "L1-sqrt", "L2", "L2-Hys"
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
			[image],
			[0, 1, 2],
			mask,
			self.bins,
			[0, 180, 0, 256, 0, 256]
		)

		hist = cv2.normalize(hist, hist).flatten()

		return hist

	def describe(self, image):
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
		segments = [
			(0, cX, 0, cY),
			(cX, w, 0, cY),
			(cX, w, cY, h),
			(0, cX, cY, h)
		]

		# Construct an elliptical mask representing the center of the image
		(axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
		ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
		cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

		features = []
		for (startX, endX, startY, endY) in segments:
			# Construct a mask for each corner of the image, subtracting
			# the elliptical center from it
			cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
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


