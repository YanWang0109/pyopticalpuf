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
    
    # @staticmethod
    # def saveImages(images: list[np.ndarray],
    #            folderToSaveImagesTo: str,
    #            sourceFolder: str = None,
    #            imageNames: list[str] = []):

    #     safePathToFolder = Path(folderToSaveImagesTo)
    #     safePathToFolder.mkdir(exist_ok=True, parents=True)

    #     # === 如果提供了原始文件夹路径，就根据它生成对应的文件名 ===
    #     if sourceFolder is not None:
    #         src = Path(sourceFolder)
    #         fileTypes = [".png", ".jpeg", ".jpg", ".webp", ".tiff", ".bmp"]
    #         filePaths = chain(*[src.glob(f"**/*{ft}") for ft in fileTypes])
    #         _imgs_and_names = [(cv2.imread(str(p)), p.name) for p in filePaths]
    #         imageNames = [name for img, name in _imgs_and_names if img is not None]

    #         # 检查数量是否匹配
    #         assert len(imageNames) == len(images), \
    #             f"数量不一致: 源文件 {len(imageNames)} vs 指纹 {len(images)}"

    #     # === 如果没有提供 imageNames，自动生成序号文件名 ===
    #     if not imageNames:
    #         imageNames = [f"image_{i}.png" for i in range(len(images))]

    #     # === 保存 ===
    #     for img, name in zip(images, imageNames):
    #         savePath = safePathToFolder / name
    #         cv2.imwrite(str(savePath), img)

    #     print(f"[DONE] Saved {len(images)} images to {safePathToFolder}")

    @staticmethod
    def saveImages(images: list[np.ndarray], folderToSaveImagesTo: str, imageNames: list[str] = []):
        safePathToFolder = Path(folderToSaveImagesTo)
        safePathToFolder.mkdir(exist_ok=True)

        if len(imageNames) == 0:
            imageNames = [f"image_{i}.png" for i in range(len(images))]

        imageAndNamePairs = zip(images, imageNames) #打包成元组
        [cv2.imwrite(f"{str(safePathToFolder / imageName)}", image) for image, imageName in imageAndNamePairs]

    @staticmethod
    def loadImagesFromFolder(pathToFolder: str, fileTypes: list[str] = [".png", ".jpeg", ".webp", ".jpg", ".bmp"]) -> list[np.ndarray]:
        """This function loads all images (in RGB) within a folder (and every subfolder) and returns them in a list."""
        safePathToFolder = Path(pathToFolder)
        ImageHelper.__checkPathIsValid(safePathToFolder)

        filePaths = chain(*[safePathToFolder.glob(f"**/*{fileType}") for fileType in fileTypes] )
        images = [cv2.imread(str(filePath)) for filePath in filePaths]
        return [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images if image is not None]