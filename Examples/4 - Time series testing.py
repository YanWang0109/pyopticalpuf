import sys
import os
OPUF_DIR = "Refactor\OPUFToolkit"
sys.path.append(os.path.dirname(OPUF_DIR))

from pyOpticalPUF.Utility import ImageHelper
from itertools import product
from pyOpticalPUF.HammingDistanceCalculators import HammingDistanceCalculator
from pyOpticalPUF.Distributions import GuassianDistribution

if __name__ == "__main__":
    intraFingerprints = ImageHelper.loadImagesFromFolder("Refactor\example output\intras")
    interFingerprints = ImageHelper.loadImagesFromFolder("Refactor\example output\inters")

    #1. Create every combination of intras (ignoring comparing the same image twice) and inters 
    allIntraCombinations = [(intraOne, intraTwo) for intraOne, intraTwo in product(intraFingerprints, intraFingerprints) if intraOne is not intraTwo]
    allIntraInterCombinations = product(intraFingerprints, interFingerprints)

    #2. Calculate hamming distance
    intraHammingDistances = [HammingDistanceCalculator.calculateHammingDistance(intraOne, intraTwo).hammingDistance for intraOne, intraTwo in allIntraCombinations]
    intersHammingDistances = [HammingDistanceCalculator.calculateHammingDistance(intra, inter).hammingDistance for intra, inter in allIntraInterCombinations]

    #3. Fit Guassian distribution and calculate metrics
    intraGuassian = GuassianDistribution.fromData(intraHammingDistances)
    interGuassian = GuassianDistribution.fromData(intersHammingDistances)

