import numpy as np
from abc import ABC, abstractmethod
from skimage.transform import resize

class FingerprintingAlgorithmParameters(ABC):
    def __post_init__(self):
        if self.__class__ == FingerprintingAlgorithmParameters:
            raise TypeError("Cannot instantiate abstract class.")

class FingerprintingAlgorithm:
    
    @abstractmethod
    def calculateFingerprint(image: np.ndarray, parameters: FingerprintingAlgorithmParameters) -> np.ndarray:
        pass
    
    @classmethod
    def _binarise(cls, image: np.ndarray, threshold: float, keySize)-> np.ndarray:
        binaryImage = (image < threshold).astype(int)
        resizedBinaryImage = resize(binaryImage, keySize, anti_aliasing=False, order=0, preserve_range=True).astype(int)
        return resizedBinaryImage.astype(np.uint8) * 255