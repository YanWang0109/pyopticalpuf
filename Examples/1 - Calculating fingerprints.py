import sys
import os
import cv2
from pathlib import Path
OPUF_DIR = ""
sys.path.append(os.path.dirname(OPUF_DIR))

from pyOpticalPUF.Utility import ImageHelper
from pyOpticalPUF.Fingerprinting.LBP import *
from pyOpticalPUF.HammingDistanceCalculators import *
from pyOpticalPUF.Metrics import *
from tqdm import tqdm


if __name__ == "__main__":
    #Set up
    intrasPath = Path("example output\Intras")
    intersPath = Path("example output\Inters")
    intrasOutputPath = Path("Refactor\example output\intras")
    intersOutputPath = Path("Refactor\example output\inters")

    intrasOutputPath.mkdir(exist_ok=True, parents=True)
    intersOutputPath.mkdir(exist_ok=True, parents=True)


    #1. Load images
    intraImages = ImageHelper.loadImagesFromFolder(intrasPath)
    intersImages = ImageHelper.loadImagesFromFolder(intersPath)

    #2 Extract single channel representation
    greyscaleIntraImages = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in intraImages]
    greyscaleIntersImages = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in intersImages]

    #3. Calculate fingerprints
    lbpParameters = LBPParameters(8, 16, (100, 100))
    intraFingerprints = [LBP.calculateFingerprint(image, lbpParameters) for image in tqdm(greyscaleIntraImages, "Calculating intra fingerprints")]
    interFingerprints = [LBP.calculateFingerprint(image, lbpParameters) for image in tqdm(greyscaleIntersImages, "Calculating inter fingerprints")]

    #4. Save fingerprints
    ImageHelper.saveImages(intraFingerprints, intrasOutputPath)
    ImageHelper.saveImages(interFingerprints, intersOutputPath)