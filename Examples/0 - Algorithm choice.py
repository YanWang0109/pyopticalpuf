from pathlib import Path
import sys
import os
import cv2
OPUF_DIR = ""
sys.path.append(os.path.dirname(OPUF_DIR))
import matplotlib.pyplot as plt


from pyOpticalPUF.Fingerprinting.AHB import *
from pyOpticalPUF.Fingerprinting.LBP import *
from pyOpticalPUF.Fingerprinting.Otsu import *
from pyOpticalPUF.Fingerprinting.Sauvola import *

if __name__ == "__main__":
    while True:
        #pathToImage = input("Enter path to image:")
        pathToImage = "/content/drive/MyDrive/cutted_square_images/a11.jpg"
        pathToImage = Path(pathToImage)
        if pathToImage.exists():
            break
        print("No image found.")
    
    image = cv2.imread(pathToImage, cv2.IMREAD_GRAYSCALE)

    # lbpParameters = LBPParameters(50, 24, (100, 100))
    # lbpFingerprint = LBP.calculateFingerprint(image, lbpParameters)
    
    sauvolaParameters = SauvolaParameters((250,250), (151,151), 0.005)
    sauvolaFingerprint = Sauvola.calculateFingerprint(image, sauvolaParameters) 

    # ahbParameters = AHBParameters((100, 100), kernelSize=35)
    # ahbFingerprint = AHB.calculateFingerprint(image, ahbParameters)

    # otsuParameters = OtsuParameters((100, 100))
    # otsuFingerprint = Ostu.calculateFingerprint(image, otsuParameters)

    plt.figure(figsize=(5,5))
    plt.imshow(sauvolaFingerprint, cmap='gray', interpolation='nearest')
    plt.title('Sauvola')
    plt.axis('off')            # 可选：去坐标轴
    plt.tight_layout()
    plt.show()

    # cv2.imshow("LBP Fingerprint", lbpFingerprint.astype(np.float32))
    # cv2.imshow("Sauvola Fingerprint", sauvolaFingerprint.astype(np.float32))
    # cv2.imshow("AHB Fingerprint", ahbFingerprint.astype(np.float32))
    # cv2.imshow("Otsu Fingerprint", otsuFingerprint.astype(np.float32))
    # cv2.waitKey(0)