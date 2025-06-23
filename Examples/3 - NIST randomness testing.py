import sys
import os

import cv2
OPUF_DIR = ""
sys.path.append(os.path.dirname(OPUF_DIR))

from pyOpticalPUF.Fingerprinting.LBP import LBP, LBPParameters
from pyOpticalPUF.Utility import ImageHelper
from pyOpticalPUF.NIST import NIST
from pyOpticalPUF.Display import FullTestingDisplay, NISTDisplay
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    intraFingerprints = ImageHelper.loadImagesFromFolder("/Users/ella/Downloads/pyOpticalPUF/example output/intras")
    interFingerprints = ImageHelper.loadImagesFromFolder("/Users/ella/Downloads/pyOpticalPUF/example output/inters")
    
    binarySequence = []
    [binarySequence.extend(np.ravel(image)) for image in intraFingerprints[:1]]

    #1. Display total bits of information
    print("Total number of bits:", len(binarySequence))

    #2. Search for duplicate segments in binary strings
    NIST.testForDuplicateSegments(binarySequence, 1000)

    testResults = NIST.runNISTTests(binarySequence)

    with open(Path("Refactor\\example output\\nistResults.csv"), 'w') as f:
        f.write("test name,result,score,time taken\n")
        for testName in testResults:
            result = testResults[testName]
            f.write(','.join(map(str, [result.name, "Passed" if result.passed else "Failed", result.score, result.timeTaken])) + '\n')

    NISTDisplay.plot(testResults)

    FullTestingDisplay([cv2.split(i)[0] for i in intraFingerprints], [cv2.split(i)[0] for i in interFingerprints], LBP, LBPParameters(10, 8, (100, 100))).plot()
