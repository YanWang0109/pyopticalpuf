import numpy as np
from .Distributions import GuassianDistribution
from statistics import mean
from math import sqrt
from scipy.integrate import quad

def __gaussianMeanCDF(distribution: GuassianDistribution, x: list[float]) -> float:
    return np.mean(distribution.cdf(x))

def TPR(intraDistribution: GuassianDistribution, intraHammingDistances: list[float]) -> float:
    return __gaussianMeanCDF(intraDistribution, intraHammingDistances)

def FNR(intraDistribution: GuassianDistribution, interHammingDistances: list[float]) -> float:
    return 1 - __gaussianMeanCDF(intraDistribution, interHammingDistances)

def TNR(interDistribution: GuassianDistribution, interHammingDistances: list[float]) -> float:
    return 1 - __gaussianMeanCDF(interDistribution, interHammingDistances)

def FPR(interDistribution: GuassianDistribution, intraHammingDistances: list[float]) -> float:
    return __gaussianMeanCDF(interDistribution, intraHammingDistances)

def decidability(guassianOne: GuassianDistribution, guassianTwo: GuassianDistribution) -> float:
    """
    Calculate decidability of two Gaussian distributions.

    Parameters:
    - guassianOne (GuassianDistribution): an instance of GuassianDistribution.
    - guassianTwo (GuassianDistribution): an instance of GuassianDistribution.

    Returns:
    - float: Decidability value.
    """    

    assert isinstance(guassianOne, GuassianDistribution), "The first parameter is not a guassian - Cannot calculate decidability from non GuassianDistribution"
    assert isinstance(guassianTwo, GuassianDistribution), "The second parameter is not a guassian - Cannot calculate decidability from non GuassianDistribution"

    absoluteDifferenceInMeans = np.abs(guassianOne.mean - guassianTwo.mean)
    averageVariance = mean([guassianOne.std**2, guassianTwo.std**2])
    squareRootOfAverageVariance = sqrt(averageVariance)
    return absoluteDifferenceInMeans / squareRootOfAverageVariance

def probabilityOfCloning(intraDisribution: GuassianDistribution, interDistribution: GuassianDistribution) -> float:
    """
    Calculate the Probability of Cloning (PoC) using Gaussian distributions.

    Parameters:
    - intraDisribution (GuassianDistribution): an instance of GuassianDistribution which corresponds to the intras.
    - interDistribution (GuassianDistribution): an instance of GuassianDistribution which corresponds to the inters.

    Returns:
    - float: Probability of Cloning (PoC).
    """
    
    assert isinstance(intraDisribution, GuassianDistribution), "The first parameter is not a guassian - Cannot calculate probability of cloning from non GuassianDistribution"
    assert isinstance(interDistribution, GuassianDistribution), "The second parameter is not a guassian - Cannot calculate probability of cloning from non GuassianDistribution"

    probabilityOfCloningValue, _ = quad(lambda x: min(intraDisribution.pdf(x), interDistribution.pdf(x)), -np.inf, np.inf)
    return probabilityOfCloningValue

def uniqueness(gaussianDistribution: GuassianDistribution):
    return gaussianDistribution.mean * 100

def reliability(gaussianDistribution: GuassianDistribution):
    return (1 - gaussianDistribution.mean) * 100

def enib(gaussianDistribution: GuassianDistribution):    
    return (gaussianDistribution.mean * (1 - gaussianDistribution.mean)) / (gaussianDistribution.std ** 2) if gaussianDistribution.std != 0 else 0