# 👇 以下是文件内容（你可以写任何代码）
import sys
import os
import cv2
from pathlib import Path
OPUF_DIR = ""
sys.path.append(os.path.dirname(OPUF_DIR))

# using Sauvola algorithm

from pyOpticalPUF.Utility import ImageHelper
from pyOpticalPUF.Fingerprinting.Sauvola import *
from pyOpticalPUF.HammingDistanceCalculators import *
from pyOpticalPUF.Metrics import *
from tqdm import tqdm
from pyOpticalPUF.Distributions import GuassianDistribution
from pyOpticalPUF.Metrics import *
from pyOpticalPUF.HammingDistanceCalculators import HammingDistanceCalculator
from pyOpticalPUF.Utility import ImageHelper
from pathlib import Path
from itertools import combinations, product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def probability_of_1(testFingerprints, save_path):
    """
    计算每张二值化图像中“1”的出现概率并作图保存，返回整体均值。
    横轴以5为间隔标注。
    """
    if len(testFingerprints) == 0:
        raise ValueError("testFingerprints 为空，无法计算。")

    save_path = Path(save_path)
    if save_path.suffix:            # 若传入具体文件名
        out_dir = save_path.parent
        out_file = save_path
    else:                           # 若传入目录
        out_dir = save_path
        out_file = out_dir / "probability_of_1.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 计算每张图像中“1”的出现概率 ===
    probs = []
    for img in testFingerprints:
        arr = np.asarray(img)
        if arr.size == 0:
            probs.append(np.nan)
            continue
        if arr.dtype == bool:
            p1 = arr.mean()
        else:
            p1 = (arr > 0).mean()
        probs.append(float(p1))

    probs = np.array(probs, dtype=float)
    mean_prob = float(np.nanmean(probs))

    # === 绘图 ===
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(probs) + 1)
    ax.plot(x, probs, marker='o', linestyle='-', linewidth=1.6)

    ax.set_xlabel("Image index")
    ax.set_ylabel("Probability of 1")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Probability of '1' per image (mean = {mean_prob:.4f})")

    # === 横轴以5为间隔标注 ===
    step = 5
    xticks = np.arange(1, len(probs) + 1, step)
    ax.set_xticks(xticks)

    # === 保存高清图像 ===
    plt.savefig(out_file, dpi=600, bbox_inches='tight')
    plt.close(fig)

    print(f"[SAVED] Figure saved to {out_file}")
    return mean_prob

if __name__ == "__main__":
    test_readpath = Path("/content/drive/MyDrive/cutted_square_images/resolution_test")
    # === 1. Load images ===
    testImages = ImageHelper.loadImagesFromFolder(test_readpath)

    # === 2. Extract single channel representation ===
    greyscaleIntraImages = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in testImages]


    # === 3. Calculate fingerprints ===
    key_size=(250, 250)
    sauvolaParameters = SauvolaParameters(key_size, (151, 151), 0.005)

    testFingerprints = [
        Sauvola.calculateFingerprint(image, sauvolaParameters)
        for image in tqdm(greyscaleIntraImages, desc="Calculating intra fingerprints")
    ]
    testOutputPath = Path("/content/drive/MyDrive/ output_results/resolution test/")
    testOutputPath = testOutputPath / str(key_size[0])
    testOutputPath.mkdir(parents=True, exist_ok=True)

    # ImageHelper.saveImages(testFingerprints, testOutputPath)

    # print(f"[DONE] Saved test to: {testOutputPath}")
    
    # testFingerprints = ImageHelper.loadImagesFromFolder("/content/drive/MyDrive/ output_results/intras/Different_irradiance_test")
                                 
    
    #1. Create every combination of intras (ignoring comparing the same image twice) and inters 
    test_pairs = list(combinations(testFingerprints, 2))


    #2. Calculate hamming distance
    testHammingDistances = [
        HammingDistanceCalculator.calculateHammingDistance(a, b).hammingDistance
        for a, b in test_pairs
    ]
    

    #3. Fit Guassian distribution and calculate metrics
    testGaussian = GuassianDistribution.fromData(testHammingDistances)
    COLOR_INTER = "#ADBED6"  # 浅蓝

    # === 绘图 ===
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    counts, bins, _ = ax.hist(
        testHammingDistances,
        bins=Bin_edges,
        color=COLOR_INTER,
        alpha=1.0,
        edgecolor="black",
        label=f'Inters  n={len(testHammingDistances)}'
    )

    # === 高斯拟合 ===
    mean_hd = np.mean(testHammingDistances)
    sigma_hd = np.std(testHammingDistances, ddof=1)
    var_hd = sigma_hd ** 2

    x = np.linspace(bins[0], bins[-1], 500)
    pdf = (1.0 / (sigma_hd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean_hd) / sigma_hd) ** 2)

    bin_width = bins[1] - bins[0]
    scale = len(testHammingDistances) * bin_width
    ax.plot(x, pdf * scale, 'k-', linewidth=2,
            label=f'Gaussian fit (μ={mean_hd:.4f}, σ²={var_hd:.4f})')

    # === 轴与标签 ===
    ax.set_xlim(0.2, 0.8)
    ax.set_xlabel("Hamming distance")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")

    # === 保存高清图像 ===
    output_file = testOutputPath / f"Hamming_histogram_key{key_size[0]}.png"
    plt.savefig(output_file, dpi=600, bbox_inches='tight')
    print(f"[SAVED] Figure saved to {output_file}")
    plt.show()
 
    mean_prob = probability_of_1(testFingerprints,testOutputPath)


   
    

