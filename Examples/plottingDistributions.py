import sys
import os
OPUF_DIR = "Refactor\OPUFToolkit"
sys.path.append(os.path.dirname(OPUF_DIR))

import matplotlib.pyplot as plt
import numpy as np
from pyOpticalPUF.Distributions import *
from pyOpticalPUF.Display import DistributionDisplay


norm = GuassianDistribution(0, 1)
norm2 = GuassianDistribution(5, 1)
xRange = list(np.arange(-5, 5, 0.01))
yRange = list(np.arange(0, 10, 0.01))
exp = ExponentialDistribution(1)


DistributionDisplay.plot(xRange, norm, norm2, exp) #labels = ["Standard Normal", "5, 1 Normal", "Standard Exponential"])
plt.title("Plotting three distributions on the same axis")
DistributionDisplay.show()
plt.clf()

DistributionDisplay.plot(xRange, norm, norm2, exp, labels = ["Standard Normal", "Normal(5,1)", "Exponential(1)"])
plt.title("Plotting three distributions on the same axis with labels")
DistributionDisplay.show()
plt.clf()

DistributionDisplay.plot3D(xRange, np.repeat(0, len(xRange)), norm)
DistributionDisplay.plot3D(xRange, np.repeat(1, len(xRange)), norm2)
DistributionDisplay.plot3D(xRange, np.repeat(2, len(xRange)), exp)
plt.title("Plotting three distributions separately on a 3D plot")
DistributionDisplay.show()
plt.clf()

DistributionDisplay.plot3D(xRange, np.repeat(0, len(xRange)), norm, label="Standard Normal")
DistributionDisplay.plot3D(xRange, np.repeat(1, len(xRange)), norm2, label="Normal(5,1)")
DistributionDisplay.plot3D(xRange, np.repeat(2, len(xRange)), exp, label="Exponential(1)")
plt.title("Plotting three distributions separately on a 3D plot with labels")
plt.legend()
DistributionDisplay.show()
plt.clf()

DistributionDisplay.plot3D(xRange, np.repeat(0, len(xRange)), norm, label="Standard Normal")
DistributionDisplay.plot3D(xRange, np.repeat(1, len(xRange)), norm2, exp, labels=["Normal(5,1)", "Exponential(1)"])
plt.title("Plotting three distributions on a 3D plot with mixed labels")
DistributionDisplay.show()
plt.clf()

normLines = DistributionDisplay.plot3D(xRange, np.repeat(0, len(xRange)), norm)
norm2Lines, expLines = DistributionDisplay.plot3D(xRange, np.repeat(1, len(xRange)), norm2, exp)
plt.legend([normLines[0], norm2Lines, expLines], ["Standard Normal", "Normal(5,1)", "Exponential(1)"])
plt.title("Plotting three distributions on a 3D plot with manual labels")
DistributionDisplay.show()
plt.clf()