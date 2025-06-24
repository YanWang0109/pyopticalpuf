from abc import abstractmethod
from itertools import product
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axisartist.parasite_axes import HostAxes

from typing import List, Literal

import numpy as np

from pyOpticalPUF.Fingerprinting.FingerprintAlgorithm import FingerprintingAlgorithm, FingerprintingAlgorithmParameters
from pyOpticalPUF.HammingDistanceCalculators import HammingDistanceCalculator
from pyOpticalPUF.Metrics import FNR, FPR, TNR, TPR, decidability, enib, probabilityOfCloning, reliability, uniqueness
from pyOpticalPUF.NIST import NIST
from .Distributions import *

LABELS = "labels"
PLOTTING_FUNCTION = "plottingFunction"

class Plotable:   
    @abstractmethod
    def plot():
        pass

    @staticmethod
    def show():
        plt.show()

class DistributionDisplay(Plotable):
    def _getPlottingFunction(dist: Distribution, plottingFunction = Literal['pdf', 'cdf']):
        if plottingFunction == "pdf":
            return dist.pdf 
        else:
            return dist.cdf

    def _validateParameters(**kwargs):       
        if not isinstance(kwargs.get("over time", False), bool):
            raise TypeError("Value provided for over time is not a boolean")
    
        labels = kwargs.get(LABELS, [])
        if not (isinstance(labels, list) and all([isinstance(item, str) for item in labels])):
            raise TypeError("Labels must be a list of strings")

        if not isinstance(kwargs.get(PLOTTING_FUNCTION, "pdf"), str):
            raise TypeError("plotting function must be a a string of either 'pdf' or 'cdf'")

        if kwargs.get(PLOTTING_FUNCTION, "pdf") not in ["pdf", "cdf"]:
            raise ValueError("Only supported plotting functions for distributions are pdf and cdf")

    @classmethod
    def plot(cls, xRange: list, *distributions: Distribution, **kwargs):
        cls._validateParameters(**kwargs)
        labels = kwargs.pop(LABELS, None)
        
        axis: Axes = kwargs.pop("ax", plt.subplot())
        plottingFunction = kwargs.pop(PLOTTING_FUNCTION, "pdf")
        lines = [axis.plot(xRange, cls._getPlottingFunction(dist, plottingFunction)(xRange), **kwargs)[0] for dist in distributions]

        if labels is not None:
            axis.legend(lines, labels)

        return lines, axis

    @classmethod
    def plot3D(cls, xRange: List, yRange: List, *distributions: Distribution, **kwargs):
        cls._validateParameters(**kwargs)
        labels = kwargs.pop(LABELS, None)
        axis: Axes = kwargs.pop("ax", None)
        if axis is None: axis = plt.subplot(projection="3d")
        plottingFunction = kwargs.pop(PLOTTING_FUNCTION, "pdf")
        lines = [axis.plot(xRange, yRange, cls._getPlottingFunction(dist, plottingFunction)(xRange),**kwargs)[0] for dist in distributions]

        if labels is not None:
            axis.legend(lines, labels)

        return lines

class NISTDisplay(Plotable):

    @staticmethod
    def reformatNISTResults( results):
        listOfResults = []
        for resultName in results:
            result = results[resultName]

            listOfResults.append([f"Passed" if result.passed else "Failed", round(result.score, 3), result.name, result.timeTaken])
        return listOfResults

    @classmethod
    def plot(cls, nistResults: dict):
        _, ax = plt.subplots(figsize=(10, 8))
        ax.axis('tight')
        ax.axis('off')

        listOfResults = cls.reformatNISTResults(nistResults)
        # Add the table
        table = ax.table(cellText=listOfResults, colLabels=['Result', 'Score', 'Test Name', 'Elapsed Time (ms)'],
                        loc='center', cellLoc='center', colWidths=[0.1, 0.1, 0.5, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)



        plt.title('NIST Test Results')
        plt.show()

class FullTestingDisplay(Plotable):
    
    def __init__(self, intras: List[np.ndarray], inters: List[np.ndarray], fingerprintingAlgorithm: FingerprintingAlgorithm, fingerprintingAlgorithmParameters: FingerprintingAlgorithmParameters):
        self.intras = intras
        self.inters = inters
        self.fingerprintingAlgorithm = fingerprintingAlgorithm
        self.fingerprintingAlgorithmParameters = fingerprintingAlgorithmParameters

    def plot(self):
        intraFingerprints = [self.fingerprintingAlgorithm.calculateFingerprint(i, self.fingerprintingAlgorithmParameters) for i in self.intras]
        interFingerprints = [self.fingerprintingAlgorithm.calculateFingerprint(i, self.fingerprintingAlgorithmParameters) for i in self.inters]

        allIntraCombinations = [(intraOne, intraTwo) for intraOne, intraTwo in product(intraFingerprints, intraFingerprints) if intraOne is not intraTwo]
        allIntraInterCombinations = product(intraFingerprints, interFingerprints)

        intraHammingDistances = [HammingDistanceCalculator.calculateHammingDistance(intraOne, intraTwo).hammingDistance for intraOne, intraTwo in allIntraCombinations]
        intersHammingDistances = [HammingDistanceCalculator.calculateHammingDistance(intra, inter).hammingDistance for intra, inter in allIntraInterCombinations]

        intraGuassian = GuassianDistribution.fromData(intraHammingDistances)
        interGuassian = GuassianDistribution.fromData(intersHammingDistances)
        reliabilityScore = reliability(intraGuassian)
        uniquenessScore = uniqueness(intraGuassian)
        enibScore = enib(interGuassian)
        decidabilityScore = decidability(intraGuassian, interGuassian)
        probabilityOfCloningScore = probabilityOfCloning(intraGuassian, interGuassian)
        uniformityScore = 0.51
        arraySize = 100
        N = 32
        decisionPoint = 0.3

        metricResultsStrings = [
            f"(Ideal:100%) Reliability: {'%.2g' % reliabilityScore}",
            f"(Ideal:50%) Uniqueness: {'%.2g' %uniquenessScore}",
            f"(Ideal:Higher) ENIB: {'%.2g' % enibScore}",
            f"(Ideal:Higher) Decidability: {'%.2g' % decidabilityScore}",
            f"(Ideal:Lower) Probability of Cloning: {'%.2g' % probabilityOfCloningScore}",
            f"(Ideal:50%) Uniformity: {'%.2g' % uniformityScore}",
            f"Array size: {arraySize}",
            f"N: {N}",
            f"Decision point: {decisionPoint}"
        ]
        # longestString = max(map(len, metricResultsStrings))
        # metricResultsStrings = [s.ljust(longestString, "*") for s in metricResultsStrings]

        mainFigure = plt.figure()
        subfigures = mainFigure.subfigures(4, 2)
        metricFigure = subfigures[1, 0]
        interIntraPlotFigure = subfigures[0, 1]
        exampleImagesFigure = subfigures[2, 0]
        nistTableFigure = subfigures[2, 1]
        experimentFigure = subfigures[2, 0]
        algorithmInfoFigure = subfigures[3, 0]


        mainFigure.text(
            x=0.5,
            y=1,
            s="pyOpticalpuf Instantaneous Testing Display", 
            horizontalalignment="center", 
            verticalalignment="top",
            fontsize="xx-large"
        )

        # Metric results
        metricFigure.text(
            x=0.1,
            y=1,
            s='\n'.join(metricResultsStrings),
            fontsize="x-large",
            verticalalignment="center",
            horizontalalignment="left"
        )        


        # Intra-inter plots
        xRange = np.arange(0, 0.5, 0.001)
        interIntraPlotAxes = interIntraPlotFigure.subplots(1,1)
        interIntraPlotAxes.plot(xRange, intraGuassian.pdf(xRange), "--", c = "orange")
        interIntraPlotAxes.hist(intraHammingDistances, bins=xRange, label = "Intra - fHD", color = "orange")
        
        interIntraPlotAxes.plot(xRange, interGuassian.pdf(xRange), "--", c = "blue")
        interIntraPlotAxes.hist(intersHammingDistances, bins=xRange, label = "Inters - fHD", color = "blue")

        interIntraPlotAxes.set_xlabel("Fractional hamming distance (fHD)")
        interIntraPlotAxes.set_ylabel("Density")

        interIntraPlotAxes.legend()
        interIntraPlotFigure.subplots_adjust(bottom=0.05)

        # Example images
        intraExamplesFigures, interExamplesFigures = exampleImagesFigure.subfigures(1, 2)
        randomInters = np.random.randint(0, len(self.inters) - 1, size=2)
        randomIntras = np.random.randint(0, len(self.intras) - 1, size=2)

        intraExampleAxes: List[Axes] = intraExamplesFigures.subplots(1, 2)
        interExampleAxes: List[Axes] = interExamplesFigures.subplots(1, 2)

        intraExamplesFigures.text(
            x=0.1,
            y=1,
            s="Intra-array example",
            fontsize="x-large",
            verticalalignment="center",
            horizontalalignment="left"
        )

        interExamplesFigures.text(
            x=0.1,
            y=1,
            s="Inter-array example",
            fontsize="x-large",
            verticalalignment="center",
            horizontalalignment="left"
        )
        [ax.set_axis_off() for ax in [*intraExampleAxes, *interExampleAxes]]
        [ax.imshow(self.intras[intraIndex], cmap=cm.get_cmap("gray")) for ax, intraIndex in zip(intraExampleAxes, randomIntras)]
        [ax.imshow(self.inters[interIndex], cmap=cm.get_cmap("gray")) for ax, interIndex in zip(interExampleAxes, randomInters)]





        # NIST
        binarySequence = []
        [binarySequence.extend(np.ravel(image)) for image in intraFingerprints[:1]]
        NIST.testForDuplicateSegments(binarySequence, 1000)
        nistResults = NISTDisplay.reformatNISTResults(NIST.runNISTTests(binarySequence))
        nistAxes = nistTableFigure.subplots(1, 1)
        nistAxes.set_axis_off()
        # Add the table
        table = nistAxes.table(cellText=nistResults, colLabels=['Result', 'Score', 'Test Name', 'Time (ms)'],
                        loc='center', cellLoc='center', colWidths=[0.1, 0.1, 0.5, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)


        # Experiment results
        experimentAxis = experimentFigure.subplots(1,1)
        experimentAxis.set_axis_off()
        experimentalTPR = '%.2g' % TPR(intraGuassian, intraHammingDistances) 
        experimentalTNR = '%.2g' % TNR(interGuassian, intersHammingDistances)
        experimentalFPR = '%.2g' % FPR(interGuassian, intraHammingDistances)
        experimentalFNR = '%.2g' % FNR(intraGuassian, intersHammingDistances)
        idealValuesRow = [1.0, 1.0, 0.0, 0.0]
        experimentResultsRow = [ experimentalTPR, experimentalTNR, experimentalFPR, experimentalFNR ]
        rows = {
            "Ideal Values": idealValuesRow,
            "Experimental": experimentResultsRow
        }
        experimentTable = experimentAxis.table(
            cellText = list(rows.values()),
            rowLabels = list(rows.keys()), 
            colLabels=["TPR", "TNR", "FPR", "FNR"],
            colWidths = [0.2, 0.2, 0.2, 0.2],
            cellLoc='center',
            )
        experimentTable.auto_set_font_size(False)
        experimentTable.set_fontsize(12)
        experimentTable.scale(1.2, 1.2)

        algorithmInfoFigure.text(
            x=0.1,
            y=0.25,
            fontsize="x-large",
            s=f"Algorithm: {self.fingerprintingAlgorithm.__name__}. Parameters: {self.fingerprintingAlgorithmParameters.__dict__}", 
            verticalalignment = "bottom"
        )

class OverTimeTestingDispaly(Plotable):

    @classmethod
    def plot(cls, intras: List[GuassianDistribution], inters: List[GuassianDistribution], intraHammings: List[List[float]], interHammings: List[List[float]]):
        fig = plt.figure("overtime figure")
        distributionPlots = fig.add_subplot(2, 2, 1, projection="3d")

        xRange = np.arange(0, 1, 0.01)
        for i, (intra, inter) in enumerate(zip(intras, inters)):
            DistributionDisplay.plot3D(xRange, np.repeat(i, xRange.shape[0]), intra, color="red", ax=distributionPlots)
            DistributionDisplay.plot3D(xRange, np.repeat(i, xRange.shape[0]), inter, color="blue", ax=distributionPlots)
        distributionPlots.set_xlabel("Hamming distance")
        distributionPlots.set_ylabel("Time")
        distributionPlots.legend(["Intras", "Inters"])


        reliabilityOverTime = [reliability(dist) for dist in intras]
        uniquenessOverTime = [uniqueness(dist) for dist in intras]
        enibOverTime = [enib(dist) for dist in intras]
        decidabilityOverTime = [enib(dist) for dist in intras]
        pocOverTime = [probabilityOfCloning(intra, inter) for intra, inter in zip(intras, inters)]

        fomPlot = fig.add_subplot(2, 2, 2)

        # Plot reliability
        fomPlot.plot(reliabilityOverTime, 'r-', label='Reliability')
        fomPlot.set_ylabel("Reliability (%)", color='r',fontsize=18)
        fomPlot.set_ylim(0, 100)
        fomPlot.tick_params(axis='y', labelcolor='r',labelsize=14)

        # Plot uniqueness
        ax2 = fomPlot.twinx()
        ax2.plot(uniquenessOverTime, 'b-', label='Uniqueness')
        ax2.set_ylabel("Uniqueness (%)", color='b',fontsize=18)
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='b',labelsize=14)

        # Plot ENIB
        ax3 = fomPlot.twinx()
        ax3.spines['right'].set_position(('outward', 70))
        ax3.plot(enibOverTime, 'g-', label='ENIB')
        ax3.set_ylabel("ENIB", color='g',fontsize=18)
        ax3.set_ylim(0, 1000)
        ax3.tick_params(axis='y', labelcolor='g',labelsize=14)

        # Plot decidability
        ax4 = fomPlot.twinx()
        ax4.spines['right'].set_position(('outward', 150))
        ax4.plot(decidabilityOverTime, 'orange', label='Decidability')
        ax4.set_ylabel("Decidability", color='orange',fontsize=18)
        ax4.tick_params(axis='y', labelcolor='orange',labelsize=14)

        # Plot PoC
        ax5 = fomPlot.twinx()
        ax5.spines['right'].set_position(('outward', 220))
        ax5.plot(pocOverTime, 'purple', label='Probability of Cloning')
        ax5.set_ylabel("PoC", color='purple',fontsize=18)
        ax5.set_yscale('log')
        ax5.tick_params(axis='y', labelcolor='purple',labelsize=14)

        # Set x-axis label and title
        fomPlot.set_xlabel("Time",fontsize=18)
        ax5.tick_params(axis='x',labelsize=14)
        fomPlot.set_title("Performance over time", fontsize=20)

        # Adding legends
        lines = [fomPlot.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0], ax4.get_lines()[0], ax5.get_lines()[0]]
        labels = [line.get_label() for line in lines]
        fomPlot.legend(lines, labels, loc='upper left')



        #Bit Error rate plot
        bitErrorPlot = fig.add_subplot(2, 2, 3)
        berOverTime = [np.mean(hds) for hds in intraHammings]
        berStdOverTime = [np.std(hds) for hds in intraHammings]

        bitErrorPlot.errorbar(range(len(berOverTime)), berOverTime, berStdOverTime)
        bitErrorPlot.set_ylabel("BER")
        bitErrorPlot.set_xlabel("Time")
        bitErrorPlot.set_title("Bit Error Rate (BER)")
        bitErrorPlot.set_ylim(0, 1)


        tprOverTime = [TPR(dist, hds) for dist, hds in zip(intras, intraHammings)]
        tnrOverTime = [TNR(dist, hds) for dist, hds in zip(inters, interHammings)]
        fprOverTime = [FPR(dist, hds) for dist, hds in zip(inters, intraHammings)]
        fnrOverTime = [FNR(dist, hds) for dist, hds in zip(intras, interHammings)]

        performancePlot = fig.add_subplot(2, 2, 4)
        # Plot FPR
        performancePlot.plot(fprOverTime, 'r-', label='FPR')
        performancePlot.set_ylabel("FPR", color='r',fontsize=18)
        performancePlot.tick_params(axis='y', labelcolor='r',labelsize=14)
        performancePlot.set_yscale('log')

        # Plot TPR
        ax2 = performancePlot.twinx()
        ax2.plot(tprOverTime, 'b-', label='TPR')
        ax2.set_ylabel("TPR", color='b',fontsize=18)
        ax2.tick_params(axis='y', labelcolor='b',labelsize=14)
        ax2.set_yscale('log')

        # Plot TNR
        ax3 = performancePlot.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(tnrOverTime, 'g-', label='TNR')
        ax3.set_ylabel("TNR", color='g',fontsize=18)
        ax3.tick_params(axis='y', labelcolor='g',labelsize=14)
        ax3.set_yscale('log')

        # Plot FNR
        ax4 = performancePlot.twinx()
        ax4.spines['right'].set_position(('outward', 120))
        ax4.plot(fnrOverTime, 'orange', label='FNR')
        ax4.set_ylabel("FNR", color='orange',fontsize=18)
        ax4.tick_params(axis='y', labelcolor='orange',labelsize=14)
        ax4.set_yscale('log')

        # Set x-axis label and title
        performancePlot.set_xlabel("Time", fontsize=14)
        performancePlot.set_title("Classification Metrics over Time", fontsize=20)

        # Adding legends
        lines = [performancePlot.get_lines()[0], ax2.get_lines()[0], ax3.get_lines()[0], ax4.get_lines()[0]]
        labels = [line.get_label() for line in lines]
        performancePlot.legend(lines, labels, loc='upper left')
        
        ...