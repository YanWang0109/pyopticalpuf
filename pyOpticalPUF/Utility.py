from itertools import chain
from pathlib import Path
import numpy as np
import cv2

class ImageHelper:
    @staticmethod
    def __checkPathIsValid(path: Path):
        """This function checks that a path is valid i.e. exists"""
        assert path.exists(), "Folder does not exist"

    @staticmethod
    def cropImages(images: list[np.ndarray], topLeft: tuple[int, int], bottomRight: tuple[int, int]):
        """This function crops a list of images to the desired coordinates."""
        return [ImageHelper.cropImage(image, topLeft, bottomRight) for image in images]

    @staticmethod
    def cropImage(image: np.ndarray, topLeft: tuple[int, int], bottomRight: tuple[int, int]):
        "This function crops an image to the desired coordinates."
        imageHeight, imageWidth, _ = image.shape
        top, left = topLeft
        bottom, right = bottomRight

        assert (-imageHeight <= top) and (top <= imageHeight), "Cannot crop from out side of the image - topLeft is not within the dimensions of the image"
        assert (-imageWidth <= left) and (left <= imageWidth), "Cannot crop from out side of the image - topLeft is not within the dimensions of the image"
        assert (-imageHeight <= bottom) and (bottom <= imageHeight), "Cannot crop from out side of the image - bottomRight is not within the dimensions of the image"
        assert (-imageWidth <= right) and (right <= imageWidth), "Cannot crop from out side of the image - bottomRight is not within the dimensions of the image"

        return image[top:bottom, left:right]

    @staticmethod
    def saveImages(images: list[np.ndarray], folderToSaveImagesTo: str, imageNames: list[str] = []):
        safePathToFolder = Path(folderToSaveImagesTo)
        safePathToFolder.mkdir(exist_ok=True)

        if len(imageNames) == 0:
            imageNames = [f"image_{i}.png" for i in range(len(images))]

        imageAndNamePairs = zip(images, imageNames)
        [cv2.imwrite(f"{str(safePathToFolder / imageName)}", image) for image, imageName in imageAndNamePairs]

    @staticmethod
    def loadImagesFromFolder(pathToFolder: str, fileTypes: list[str] = [".png", ".jpeg", ".webp", ".jpg", ".bmp"]) -> list[np.ndarray]:
        """This function loads all images (in RGB) within a folder (and every subfolder) and returns them in a list."""
        safePathToFolder = Path(pathToFolder)
        ImageHelper.__checkPathIsValid(safePathToFolder)

        filePaths = chain(*[safePathToFolder.glob(f"**/*{fileType}") for fileType in fileTypes] )
        images = [cv2.imread(str(filePath)) for filePath in filePaths]
        return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images if image is not None]