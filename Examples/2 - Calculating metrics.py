import sys
import os
OPUF_DIR = ""
sys.path.append(os.path.dirname(OPUF_DIR))

from pyOpticalPUF.Distributions import GuassianDistribution
from pyOpticalPUF.Metrics import *
from pyOpticalPUF.HammingDistanceCalculators import HammingDistanceCalculator
from pyOpticalPUF.Utility import ImageHelper
from pathlib import Path
from itertools import combinations, product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

if __name__ == "__main__":
    intraFingerprints = ImageHelper.loadImagesFromFolder("/content/drive/MyDrive/ output_results/intras/Different_irradiance")
    interRootDir = Path("/content/drive/MyDrive/ output_results/inters") 
    inter_labels = sorted([d for d in interRootDir.iterdir() if d.is_dir()])
    if len(inter_labels) < 2:
        raise RuntimeError(f"Need at least 2 labels in {interRootDir}, got: {inter_labels}")

    interA_fps = ImageHelper.loadImagesFromFolder(inter_labels[0])
    interB_fps = ImageHelper.loadImagesFromFolder(inter_labels[1])
    print(f"[INFO] inter labels: {inter_labels[0].name} ({len(interA_fps)}), {inter_labels[1].name} ({len(interB_fps)})")

    #1. Create every combination of intras (ignoring comparing the same image twice) and inters 
    intra_pairs = list(combinations(intraFingerprints, 2))
    inter_pairs = list(product(interA_fps, interB_fps))

    #2. Calculate hamming distance
    intraHammingDistances = [
        HammingDistanceCalculator.calculateHammingDistance(a, b).hammingDistance
        for a, b in intra_pairs
    ]
    interHammingDistances = [
        HammingDistanceCalculator.calculateHammingDistance(a, b).hammingDistance
        for a, b in inter_pairs
    ]
    
    intraDistanceMap = {}
    for (a, b), dist in zip(intra_pairs, intraHammingDistances):
     nameA = getattr(a, "filename", None) or getattr(a, "path", "unknown_A")
     nameB = getattr(b, "filename", None) or getattr(b, "path", "unknown_B")
     intraDistanceMap[(nameA, nameB)] = dist

    #3. Fit Guassian distribution and calculate metrics
    intraGaussian = GuassianDistribution.fromData(intraHammingDistances)
    interGaussian = GuassianDistribution.fromData(interHammingDistances)
    reliabilityScore          = reliability(intraGaussian)                         # 稳定性（越高越好）
    uniquenessScore           = uniqueness(interGaussian)                          # 唯一性（越高越好）
    enibScore                 = enib(interGaussian)                                # 参考你项目定义
    decidabilityScore         = decidability(intraGaussian, interGaussian)         # 可分性 d'
    probabilityOfCloningScore = probabilityOfCloning(intraGaussian, interGaussian) # 克隆概率（越低越好）


    print(f"""Metrics for Fluorescent Material:
Reliability: {reliabilityScore}%
Uniqueness: {uniquenessScore}%
ENIB: {enibScore}
Decidability: {decidabilityScore}
Probability of cloning: {probabilityOfCloningScore}""")

    COLOR_INTRA = "#D5E4A6"  # 浅绿
    COLOR_INTER = "#ADBED6"  # 浅蓝

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    Bin_edges = np.arange(0, 1.01, 0.01)
    ax.hist(intraHammingDistances, bins=Bin_edges, color=COLOR_INTRA, alpha=1.0,
            edgecolor="black", label=f"Intras(diff illumination) n={len(intraHammingDistances)}")
    ax.hist(interHammingDistances, bins=Bin_edges, color=COLOR_INTER, alpha=1.0,
            edgecolor="black", label=f"Inters n={len(interHammingDistances)}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Hamming distance")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    plt.show()

    # # 初始阈值
    # ymin, ymax = ax.get_ylim()
    # init_thresh = 0.30
    # thresholdLine = ax.vlines(init_thresh, ymin, ymax, colors="red", linestyles="dashed", label="Threshold")

    # # 滑块
    # slider_ax = fig.add_axes([0.15, 0.08, 0.65, 0.03])
    # thresholdSlider = Slider(slider_ax, "Threshold", 0.01, 0.5, valinit=init_thresh)

    # def updatePlot(_=None):
    #     thr = thresholdSlider.val

    #     # TPR: intra <= thr 判为“同一样品”算对
    #     TPR = float(np.mean(intraHammingDistances <= thr)) * 100.0

    #     # TNR: inter > thr 判为“不同样品”算对
    #     TNR = float(np.mean(interHammingDistances >  thr)) * 100.0

    #     FNR = 100.0 - TPR
    #     FPR = 100.0 - TNR

    #     # 更新阈值线位置
    #     ymin, ymax = ax.get_ylim()
    #     thresholdLine.set_segments([np.array([[thr, ymin], [thr, ymax]])])

    #     ax.set_title(f"TPR: {TPR:.2f}%,  TNR: {TNR:.2f}%\nFNR: {FNR:.2f}%,  FPR: {FPR:.2f}%")
    #     fig.canvas.draw_idle()

    # thresholdSlider.on_changed(updatePlot)
    # updatePlot()  # 初始化标题

    # plt.tight_layout(rect=[0, 0.12, 1, 1])
# if __name__ == "__main__":
#     intraFingerprints = ImageHelper.loadImagesFromFolder("Refactor\example output\intras")
#     interFingerprints = ImageHelper.loadImagesFromFolder("Refactor\example output\inters")

#     #1. Create every combination of intras (ignoring comparing the same image twice) and inters 
#     allIntraCombinations = [(intraOne, intraTwo) for intraOne, intraTwo in product(intraFingerprints, intraFingerprints) if intraOne is not intraTwo]
#     allIntraInterCombinations = product(intraFingerprints, interFingerprints)

#     #2. Calculate hamming distance
#     intraHammingDistances = [HammingDistanceCalculator.calculateHammingDistance(intraOne, intraTwo).hammingDistance for intraOne, intraTwo in allIntraCombinations]
#     intersHammingDistances = [HammingDistanceCalculator.calculateHammingDistance(intra, inter).hammingDistance for intra, inter in allIntraInterCombinations]

#     #3. Fit Guassian distribution and calculate metrics
#     intraGuassian = GuassianDistribution.fromData(intraHammingDistances)
#     interGuassian = GuassianDistribution.fromData(intersHammingDistances)
#     reliabilityScore = reliability(intraGuassian)
#     uniquenessScore = uniqueness(intraGuassian)
#     enibScore = enib(interGuassian)
#     decidabilityScore = decidability(intraGuassian, interGuassian)
#     probabilityOfCloningScore = probabilityOfCloning(intraGuassian, interGuassian)

#     print(f"""Metrics for Fluorescent Material:
# Reliability: {reliabilityScore}%
# Uniqueness: {uniquenessScore}%
# ENIB: {enibScore}
# Decidability: {decidabilityScore}
# Probability of cloning: {probabilityOfCloningScore}""")

#     fig, axes = plt.subplots()
#     intraLine = axes.hist(intraHammingDistances, bins = 10, label=f"Intras n={len(intraHammingDistances)}", alpha = 0.5)
#     interLine = axes.hist(intersHammingDistances, bins = 10, label=f"Inters n={len(intersHammingDistances)}", alpha = 0.5)
#     axes.set_xlim(0, 0.5)
#     thresholdLine = axes.vlines(0.3, plt.ylim()[0], plt.ylim()[1], colors="red", linestyles="dashed", label="Threshold")
#     hammingAxes = fig.add_axes([0.15, 0.025, 0.65, 0.03])
#     thresholdSlider = Slider(hammingAxes, "Threshold", 0.01, 0.5, valinit=0.3)

#     def updatePlot(_):
#         truePositiveRate = round(mean([1 if intra <= thresholdSlider.val else 0 for intra in intraHammingDistances]) * 100, 2)
#         trueNegativeRate = round(mean([1 if inter > thresholdSlider.val else 0 for inter in intersHammingDistances]) * 100, 2)

#         oldSegments = thresholdLine.get_segments()

#         ymin = oldSegments[0][0, 1]
#         ymax = oldSegments[0][1, 1]

#         newSegments = [
#             [thresholdSlider.val, ymin],
#             [thresholdSlider.val, ymax]
#         ]

#         thresholdLine.set_segments([np.array(newSegments)])

#         axes.set_title(f"TPR: {truePositiveRate}%, TNR: {trueNegativeRate}%\nFNR: {100-truePositiveRate}%, FPR: {100-trueNegativeRate}%")
#         fig.canvas.draw_idle()

#     thresholdSlider.on_changed(updatePlot)
#     axes.legend()
#     plt.show()