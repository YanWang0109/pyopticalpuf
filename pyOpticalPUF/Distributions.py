from dataclasses import dataclass
from scipy.stats import norm, expon
from statistics import mean, stdev

class Distribution:
    pass

@dataclass
class GuassianDistribution(Distribution):
    mean: float
    std: float
    
    def pdf(self, x):
        return norm.pdf(x, self.mean, self.std)
    
    def cdf(self, x):
        return norm.cdf(x, loc=self.mean, scale=self.std)
    
    @staticmethod
    def fromData(data: list):
        return GuassianDistribution(mean(data), stdev(data))

@dataclass
class ExponentialDistribution(Distribution):
    rate: float

    def pdf(self, x):
        return expon.pdf(x, self.rate)
    
    def cdf(self, x):
        return expon.cdf(x, self.rate)