import numpy as np
from dataclasses import dataclass

@dataclass
class HammingResult:
    """A dataclass containing the hamming mask and corresponding hammingDistance of a hamming distance calculation operation"""
    hammingMask: np.ndarray
    hammingDistance: float

class HammingDistanceCalculator:
    @classmethod
    def __calculateHammingMask(cls, *images: np.ndarray) -> np.ndarray:
        """This function calculates the hamming mask by performing a logical xor on all input images"""
        images = images[0]
        # assert all([image.ndim == 1 for image in images]), "Cannot calculate hamming distance from image with more than one channel"

        #TODO: add assert in to ensure that the images are binary images
        hammingMask = images[0]
        for i in range(1, len(images)):
            hammingMask = np.logical_xor(hammingMask, images[i])

        return hammingMask
    @classmethod
    def calculateHammingDistance(cls, *images: np.ndarray) -> HammingResult:
        """This function calculates the hamming distance of a set of binary images"""
        hammingMask = cls.__calculateHammingMask(images)
        return HammingResult(hammingMask, float(hammingMask.mean()))