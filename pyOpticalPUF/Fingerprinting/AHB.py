from numpy import ndarray
from .FingerprintAlgorithm import *
from dataclasses import dataclass

@dataclass
class AHBParameters(FingerprintingAlgorithmParameters):
    keySize: tuple[int, int]
    kernelSize: int = 36

class AHB(FingerprintingAlgorithm):
    """
    Adaptive High Boost (AHB) algorithm for binarization of grayscale images.

    Parameters:
    - image (ndarray): Input grayscale image data.
    - parameters (AHBParameters): Parameters of AHB

    Returns:
    - binary_image_resized (ndarray): Binarized image resized to keysize x keysize.
    """

    @classmethod
    def calculateFingerprint(cls, singleChannelImage: ndarray, parameters: AHBParameters) -> ndarray:
        assert isinstance(parameters, AHBParameters), "Provided parameters are not AHB parameters"
        assert parameters.kernelSize % 2 == 1, "Kernel size must be odd"

        n = parameters.kernelSize
        kernel = np.ones((n, n), dtype=np.float32)
        kernel[(n//2), (n//2)] = -n**2 + 1

        filtered_image = np.zeros_like(singleChannelImage, dtype=np.uint8)

        height, width = singleChannelImage.shape
        for i in range(height - n + 1):
            for j in range(width - n + 1):
                window = singleChannelImage[i:i+n, j:j+n].copy()
                result = np.sum(window * kernel)
                if result < 0:
                    filtered_image[i, j] = 0  # Black
                else:
                    filtered_image[i, j] = 255  # White


        return filtered_image