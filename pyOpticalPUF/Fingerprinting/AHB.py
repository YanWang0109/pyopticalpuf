from numpy import ndarray
from .FingerprintAlgorithm import *
from dataclasses import dataclass

@dataclass
class AHBParameters(FingerprintingAlgorithmParameters):
    keySize: tuple[int, int]
    kernelSize: int = 35

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

        # Define the kernel
        n = parameters.kernelSize
        kernel = np.ones((n, n), dtype=np.float32)
        kernel[(n//2), (n//2)] = -n**2 + 1

    # Output shape after valid convolution

        outputRows = singleChannelImage.shape[0] - n + 1
        outputCols = singleChannelImage.shape[1] - n + 1
        filteredImage = np.zeros((outputRows, outputCols), dtype=np.uint8)

        for i in range(outputRows):
            for j in range(outputCols):
                window = singleChannelImage[i:i+n, j:j+n]
                filteredImage[i, j] = -np.sum(window * kernel)

        return cls._binarise(filteredImage, 0, parameters.keySize)