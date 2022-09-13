#%% SINGLE VARIABLE

import numpy as np
import scipy
from scipy.stats import iqr, tvar, tstd
var1 = np.random.randn(100)

# FIVE NUM

var1.mean()
np.mean(var1)


np.median(var1)
np.median(var1)


var1.min()
var1.max()


# RANGE
np.ptp(var1, axis=0)


# IQR

iqr(var1)


# VARIANCE

tvar(var1)
np.var(var1)

# STANDARD DEVIATION
tstd(var1)
np.std(var1)

# GEOMETRIC MEAN
var2 = [7,4,7,9,2,4,3,5,6,4]
scipy.stats.gmean(var2)

# HARMONIC MEAN
scipy.stats.hmean(var2)


# MODE
scipy.stats.mode(var2)


# QUANTILES
scipy.percentile(var1, q = np.linspace(0, 1, num=10))
np.quantile(var1,q = np.linspace(0, 1, num=10), axis=0)
# SKEWNESS
scipy.stats.skew(var1)

# KURTOSIS
scipy.stats.kurtosis(var1)


# VARIATION COEFFICIENT

scipy.stats.variation(var1)

# CENTRAL MOMENTS
scipy.stats.moment(var1, moment = 2)

#%% TWO VARIABLES

x = np.random.randn(100)
y = np.random.randn(100)

# COVARIANCE

np.cov(x,y)

# CORRELATION

scipy.stats.pearsonr(x, y)
scipy.stats.spearmanr(x, y)
scipy.stats.kendalltau(x, y)








