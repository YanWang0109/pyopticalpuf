import sys
import os
OPUF_DIR = ""
sys.path.append(os.path.dirname(OPUF_DIR))

from pyOpticalPUF.Distributions import GuassianDistribution
from pyOpticalPUF.Metrics import *
from pyOpticalPUF.HammingDistanceCalculators import HammingDistanceCalculator
from pyOpticalPUF.Utility import ImageHelper
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from itertools import product
from statistics import mean

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
    reliabilityScore = reliability(intraGuassian)
    uniquenessScore = uniqueness(intraGuassian)
    enibScore = enib(interGuassian)
    decidabilityScore = decidability(intraGuassian, interGuassian)
    probabilityOfCloningScore = probabilityOfCloning(intraGuassian, interGuassian)

    print(f"""Metrics for Fluorescent Material:
Reliability: {reliabilityScore}%
Uniqueness: {uniquenessScore}%
ENIB: {enibScore}
Decidability: {decidabilityScore}
Probability of cloning: {probabilityOfCloningScore}""")

    fig, axes = plt.subplots()
    intraLine = axes.hist(intraHammingDistances, bins = 10, label=f"Intras n={len(intraHammingDistances)}", alpha = 0.5)
    interLine = axes.hist(intersHammingDistances, bins = 10, label=f"Inters n={len(intersHammingDistances)}", alpha = 0.5)
    axes.set_xlim(0, 0.5)
    thresholdLine = axes.vlines(0.3, plt.ylim()[0], plt.ylim()[1], colors="red", linestyles="dashed", label="Threshold")
    hammingAxes = fig.add_axes([0.15, 0.025, 0.65, 0.03])
    thresholdSlider = Slider(hammingAxes, "Threshold", 0.01, 0.5, valinit=0.3)

    def updatePlot(_):
        truePositiveRate = round(mean([1 if intra <= thresholdSlider.val else 0 for intra in intraHammingDistances]) * 100, 2)
        trueNegativeRate = round(mean([1 if inter > thresholdSlider.val else 0 for inter in intersHammingDistances]) * 100, 2)

        oldSegments = thresholdLine.get_segments()

        ymin = oldSegments[0][0, 1]
        ymax = oldSegments[0][1, 1]

        newSegments = [
            [thresholdSlider.val, ymin],
            [thresholdSlider.val, ymax]
        ]

        thresholdLine.set_segments([np.array(newSegments)])

        axes.set_title(f"TPR: {truePositiveRate}%, TNR: {trueNegativeRate}%\nFNR: {100-truePositiveRate}%, FPR: {100-trueNegativeRate}%")
        fig.canvas.draw_idle()

    thresholdSlider.on_changed(updatePlot)
    axes.legend()
    plt.show()