#%% Sampling

import random
import numpy as np

## Seed

np.random.seed(111)
random.seed(111)

setA = list(range(1,11))

# Generate:
#  1. Permutation without replacement
random.sample(setA, k =10)
setB = random.shuffle(setA) ## !!! shuffle inplace !!!
np.random.permutation(setA)

#  2. Permutation with replacement
random.choices(setA, k=10)

#  3. Variation without replacement
random.sample(setA, k =4)

#  4. Variation with replacement
random.choices(setA, k=4)


# Random single integer from seqence:

random.choice(setA) # single choice from given set
random.randrange(100) # range from 1 to 100
random.randrange(10,12) # range from 10 to 11
random.randint(10, 11) # same as random.randrange(a, b+1)
np.random.choice(setA)

# Sequence of random integers:
np.random.choice(setA, size = 4, replace=False)
random.choices(setA, k=10)

# Random single float [0,1]:

random.random()
random.uniform(0,1)

# Sequence of random floats:
np.random.rand(10) 
np.random.random(10)
np.random.sample(10)

# random sample with given probabilities:
random.choices(['A', 'B', 'C'], weights = [0.1,0.2,0.6], k = 20)

# random matrix
np.random.rand(3, 3) 



#%% CONTINUOUS DISTRIBUTIONS

from scipy import stats
import matplotlib.pyplot as plt 

## pdf: density                 #### pdf(x, loc=0, scale=1)
## cdf: cumulative probability  #### cdf(x, loc=0, scale=1)
## ppf: quantile                #### ppf(q, loc=0, scale=1)
## rvs: random                  #### rvs(loc=0, scale=1, size=1, random_state=None)

## UNIFORM
x = range(0,100)
# density
y = stats.uniform.pdf(range(0,100), loc= 0, scale = 100)
plt.plot(x,y)

# cumulative distribution
y = stats.uniform.cdf(range(0,100), loc= 0, scale = 100)
plt.plot(x,y)

# quantile
stats.uniform.ppf(0.5, loc= 0, scale = 100)

# random
y = stats.uniform.rvs(size = 1000, loc= 0, scale = 100)
plt.hist(y)



## NORMAL
x = np.linspace(-4,4, num = 100)

# density
y = stats.norm.pdf(x, loc = 0, scale = 2) 
plt.plot(x,y)

# cumulative distribution
y = stats.norm.cdf(x, loc = 0, scale = 2) 
plt.plot(x,y)

# quantile
stats.norm.ppf(0.5, loc = 0, scale = 2) 
# random
y = stats.norm.rvs(size = 100, loc = 0, scale = 2) 
plt.hist(y)

## NUMPY !!!!
np.random.randn(100)# random sample from normal dist
np.random.randn(5,5)# matrix 5,5 with random sample from normal dist
# or RANDOM
random.gauss(2,1)




## LOGNORMAL

# density
stats.lognorm.pdf
# cumulative distribution
stats.lognorm.cdf
# quantile
stats.lognorm.ppf
# random
stats.lognorm.rvs


## CHI-SQ

# density
stats.chi2.pdf
# cumulative distribution
stats.chi2.cdf
# quantile
stats.chi2.ppf
# random
stats.chi2.rvs


## EXPONENTIAL

# density
stats.expon.pdf
# cumulative distribution
stats.expon.cdf
# quantile
stats.expon.ppf
# random
stats.expon.rvs


## GAMMA
# density
stats.gamma.pdf
# cumulative distribution
stats.gamma.cdf
# quantile
stats.gamma.ppf
# random
stats.gamma.rvs


## BETA
# density
stats.beta.pdf
# cumulative distribution
stats.beta.cdf
# quantile
stats.beta.ppf
# random
stats.beta.rvs



## t-Student
# density
stats.t.pdf(3,4)
# cumulative distribution
stats.t.cdf
# quantile
stats.t.ppf
# random
stats.t.rvs

## F-Snedecor

# density
stats.f.pdf
# cumulative distribution
stats.f.cdf
# quantile
stats.f.ppf
# random
stats.f.rvs


##Cauchy
# density
stats.cauchy.pdf
# cumulative distribution
stats.cauchy.cdf
# quantile
stats.cauchy.ppf
# random
stats.cauchy.rvs



## Weibull
# density
stats.weibull_min.pdf
# cumulative distribution
stats.weibull_min.cdf
# quantile
stats.weibull_min.ppf
# random
stats.weibull_min.rvs


################################
## DISCRETE DISTRIBUTIONS   ####
################################

## pmf: mass function           #### pmf(k, n, p, loc=0)
## cdf: cumulative probability  #### cdf(k, n, p, loc=0)
## ppf: quantile                #### ppf(q, n, p, loc=0)
## rvs: random                  #### rvs(n, p, loc=0, size=1, random_state=None)



## BERNOULLI

x = [0,1]
# density
y = stats.bernoulli.pmf(k = x, p = 0.3)
plt.scatter(x, y)
# cumulative distribution
y = stats.bernoulli.cdf(k = x, p = 0.3)
plt.scatter(x, y)
# quantile
stats.bernoulli.ppf(0.5, p = 0.3)
# random
y = stats.bernoulli.rvs(p = 0.3, size = 100)
plt.hist(y)



## BINOMIAL
x = range(0, 21) # number of success
# density
y = stats.binom.pmf(k = x, p = 0.3, n = 20)
plt.scatter(x, y)

# cumulative distribution
y = stats.binom.cdf(k = x, p = 0.3, n = 20)
plt.scatter(x, y)


# quantile
stats.binom.ppf(q = 0.25, p = 0.3, n = 20)
# random
y = stats.binom.rvs(size = 1000, p = 0.3, n = 20)
plt.hist(y)



# POISSON
x = range(0,25)
# density
y = stats.poisson.pmf(k = x, mu = 5)
plt.scatter(x, y)

# cumulative distribution
y = stats.poisson.cdf(k = x, mu = 5)
plt.scatter(x, y)

# quantile
stats.poisson.ppf(q = 0.9, mu = 5)
# random
y = stats.poisson.rvs(size = 1000, mu = 5)
plt.hist(y)


## GEOMETRIC

x = range(0,40)
# density
y = stats.geom.pmf(k = x, p = 0.2)
plt.scatter(x, y)

# cumulative distribution
y = stats.geom.cdf(k = x, p = 0.2)
plt.scatter(x, y)

# quantile
stats.geom.ppf(q = 0.25, p = 0.2)
# random
y = stats.geom.rvs(size = 100, p = 0.2)
plt.hist(y)


## HYPERGEOMETRIC
x = range(0,20)

# density
y = stats.hypergeom.pmf(k = x, M = 100,n = 40,N = 10)
plt.scatter(x, y)

# cumulative distribution
y = stats.hypergeom.cdf(k = x, M = 100,n = 40,N = 10)
plt.scatter(x, y)

# quantile
stats.hypergeom.ppf(q = 0.9, M = 100,n = 40,N = 10 )
# random
y = stats.hypergeom.rvs(size = 1000,  M = 100,n = 40,N = 10)
plt.hist(y)



################################
## CENTRAL LIMIT THEOREM    ####
################################

v = []
for i in range(10000):
  v.append(np.mean(stats.norm.rvs(size = 100, loc = 0, scale = 1)))

plt.hist(v)

v = []
for i in range(10000):
  v.append(np.mean(stats.poisson.rvs(size=10000, mu=5)))

plt.hist(v)


v_stand = (v-np.mean(v))/np.std(v)

plt.hist(v_stand)
np.mean(v_stand)
np.std(v_stand)

