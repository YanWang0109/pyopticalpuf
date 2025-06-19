from dataclasses import dataclass
from numpy import ndarray
from .FingerprintAlgorithm import *
from skimage.filters import threshold_sauvola

@dataclass
class SauvolaParameters:
    keySize: tuple[int, int]
    windowSize: tuple[int, int]
    k: float
    r: float = None

class Sauvola(FingerprintingAlgorithm):
    """
    Sauvola Binarization algorithm for generating a binary map from a grayscale image.

    Parameters:
    - singleChannelImage (ndarray): Input grayscale image data.
    - parameters (SauvolaParameters): Parameters for sauvola

    Returns:
    - binary_map_resized (ndarray): 2D binary map resized to keysize x keysize.
    """

    @classmethod
    def calculateFingerprint(cls, singleChannelImage: ndarray, parameters: SauvolaParameters) -> np.ndarray:
        assert isinstance(parameters, SauvolaParameters), "Provided parameters are not sauvolda parameters"
        sauvoldaThresholdedImage = threshold_sauvola(singleChannelImage, parameters.windowSize, parameters.k, parameters.r)

        return cls._binarise(singleChannelImage, sauvoldaThresholdedImage, parameters.keySize)