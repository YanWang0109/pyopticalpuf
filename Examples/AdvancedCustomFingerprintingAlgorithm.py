import sys
import os
import cv2
from numpy import ndarray
from dataclasses import dataclass
OPUF_DIR = "Refactor\OPUFToolkit"
sys.path.append(os.path.dirname(OPUF_DIR))

from pyOpticalPUF.Fingerprinting.FingerprintAlgorithm import FingerprintingAlgorithmParameters, FingerprintingAlgorithm

@dataclass
class CustomFingerprintAlgorithmParameters(FingerprintingAlgorithmParameters):
    lowerBound: float
    upperBound: float

class CustomFingerprintingAlgorithm(FingerprintingAlgorithm):

    def calculateFingerprint(singleChannelImage: ndarray, parameters: CustomFingerprintAlgorithmParameters) -> ndarray:
        return cv2.inRange(singleChannelImage, parameters.lowerBound, parameters.upperBound)