import sys
import os
import cv2
from pathlib import Path
OPUF_DIR = ""
sys.path.append(os.path.dirname(OPUF_DIR))

# using Sauvola algorithm 

from pyOpticalPUF.Utility import ImageHelper
#from pyOpticalPUF.Fingerprinting.LBP import *
from pyOpticalPUF.Fingerprinting.Sauvola import *
from pyOpticalPUF.HammingDistanceCalculators import *
from pyOpticalPUF.Metrics import *
from tqdm import tqdm


if __name__ == "__main__":
    #Set up
    intrasPath = Path("/content/drive/MyDrive/cutted_square_images/intra/identical /a11")
    intersRoot       = Path("/content/drive/MyDrive/cutted_square_images/inter")
    intrasOutputPath = Path("/content/drive/MyDrive/ output_results/intras/identical /a11")
    intersOutputPath = Path("/content/drive/MyDrive/ output_results/inters")

    intrasOutputPath.mkdir(exist_ok=True, parents=True)
    intersOutputPath.mkdir(exist_ok=True, parents=True)


    #1. Load images
    intraImages = ImageHelper.loadImagesFromFolder(intrasPath)
    label_to_grayimgs = {}
    for sub in sorted(p for p in intersRoot.iterdir() if p.is_dir()):
        imgs = ImageHelper.loadImagesFromFolder(sub)
        label_to_grayimgs[sub.name] = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
        print(f"[INFO] {sub.name}: loaded {len(imgs)} images")

    #2 Extract single channel representation
    greyscaleIntraImages = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in intraImages]
   
    #3. Calculate fingerprints
    sauvolaParameters = SauvolaParameters((250,250), (151,151), 0.005)
    intraFingerprints = [Sauvola.calculateFingerprint(image, sauvolaParameters) for image in tqdm(greyscaleIntraImages, "Calculating intra fingerprints")]
    label_to_fps = {}
    for label, gs_imgs in label_to_grayimgs.items():
        fps = [
            Sauvola.calculateFingerprint(img, sauvolaParameters)
            for img in tqdm(gs_imgs, desc=f"Calculating inter fingerprints [{label}]")
        ]
        label_to_fps[label] = fps


    #4. Save fingerprints
    ImageHelper.saveImages(intraFingerprints, intrasOutputPath)

    for label, fps in label_to_fps.items():
        out_dir = intersOutputPath / label
        out_dir.mkdir(parents=True, exist_ok=True)
        ImageHelper.saveImages(fps, out_dir)

    print("[DONE] Saved intra to:", intrasOutputPath)
    print("[DONE] Saved inter (by label) to:", intersOutputPath, "->", list(label_to_fps.keys()))


# if __name__ == "__main__":
#     #Set up
#     intrasPath = Path("/content/drive/MyDrive/cutted_square_images/intra/identical /a11")
#     intersPath = Path("/content/drive/MyDrive/cutted_square_images/inter")
#     intrasOutputPath = Path("/content/drive/MyDrive/ output/intras")
#     intersOutputPath = Path("/content/drive/MyDrive/ output/inters")

#     intrasOutputPath.mkdir(exist_ok=True, parents=True)
#     intersOutputPath.mkdir(exist_ok=True, parents=True)


#     #1. Load images
#     intraImages = ImageHelper.loadImagesFromFolder(intrasPath)
#     intersImages = ImageHelper.loadImagesFromFolder(intersPath)

#     #2 Extract single channel representation
#     greyscaleIntraImages = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in intraImages]
#     greyscaleIntersImages = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in intersImages]

#     #3. Calculate fingerprints
#     lbpParameters = LBPParameters(8, 16, (100, 100))
#     intraFingerprints = [LBP.calculateFingerprint(image, lbpParameters) for image in tqdm(greyscaleIntraImages, "Calculating intra fingerprints")]
#     interFingerprints = [LBP.calculateFingerprint(image, lbpParameters) for image in tqdm(greyscaleIntersImages, "Calculating inter fingerprints")]

#     #4. Save fingerprints
#     ImageHelper.saveImages(intraFingerprints, intrasOutputPath)
#     ImageHelper.saveImages(interFingerprints, intersOutputPath)

