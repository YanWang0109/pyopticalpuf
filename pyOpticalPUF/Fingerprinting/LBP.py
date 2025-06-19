from numpy import ndarray
from .FingerprintAlgorithm import *
from dataclasses import dataclass
from skimage.feature import local_binary_pattern
from enum import Enum

class LBPMethod(Enum):
    DEFAULT = "default"
    ROR = "ror"
    UNIFORM = "uniform"
    VAR = "var"

@dataclass
class LBPParameters(FingerprintingAlgorithmParameters):
    radius: int
    neighbours: int
    keySize: tuple[int, int]
    method: LBPMethod = LBPMethod.UNIFORM

class LBP(FingerprintingAlgorithm):
    """
    Local Binary Pattern (LBP) algorithm for generating a binary map from an image.

    Parameters:
    - image (ndarray): Input image data. It should be in grayscale format.
    - parameters (LBPParameters): Parameters for LBP

    Returns:
    - binary_map_resized (ndarray): 2D binary map resized to keysize x keysize.
    """

    @classmethod
    def calculateFingerprint(cls, singleChannelImage: ndarray, parameters: LBPParameters) -> np.ndarray:
        assert isinstance(parameters, LBPParameters), "Provided parameters are not LBP parameters"
        localBinaryPatterns = local_binary_pattern(singleChannelImage, parameters.neighbours, parameters.radius, parameters.method.value)
        return cls._binarise(localBinaryPatterns, np.mean(localBinaryPatterns), parameters.keySize)