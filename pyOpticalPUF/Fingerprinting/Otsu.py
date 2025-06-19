from dataclasses import dataclass
from skimage.filters import threshold_otsu
from numpy import ndarray
from .FingerprintAlgorithm import *

@dataclass
class OtsuParameters:
    keySize: tuple[int, int]

class Ostu(FingerprintingAlgorithm):
    """
    Apply Otsu's thresholding method to binarize a grayscale image.

    Parameters:
    - image (ndarray): Input grayscale image data.
    - parameters (OtsuParameters): Parameters for Otsu.

    Returns:
    - binary_image_resized (ndarray): Binarized image resized to keysize x keysize.
    """
    @classmethod
    def calculateFingerprint(cls, singleChannelImage: ndarray, parameters: OtsuParameters) -> ndarray:
        assert isinstance(parameters, OtsuParameters), "Provided parameters are not Otsu parameters"
        otsuThresholdedImage = threshold_otsu(singleChannelImage)

        return cls._binarise(singleChannelImage, otsuThresholdedImage, parameters.keySize)