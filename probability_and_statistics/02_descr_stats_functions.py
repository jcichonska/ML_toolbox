import numpy as np

v = np.random.randn(100)


#%% MEAN

def cust_mean(vec):
    return vec.sum()/len(vec)

cust_mean(v)
v.mean()


#%% weighted mean

w = np.random.randint(1,10,size = 100)

def cust_mean(vec, w):
    return np.dot(v,w)/len(vec)

cust_mean(v, w)


## VARIANCE
def cust_var(vec):
    return ((vec-vec.mean())**2).sum()/(len(vec)-1)

cust_var(v)
v.var()

## STD
def cust_std(vec):
    return (((vec-vec.mean())**2).sum()/len(vec))**(1/2)

cust_std(v)
v.std()


## EXPECTED VALUE
def expected_value(vec):
   n = vec.shape[0]
   w = [1/n]*n
   return np.dot(vec,w)

expected_value(v)    
v.mean()  
  
## COVARIANCE
v1 = np.random.randint(0,10, size = 500)
v2 = np.random.randint(0,10, size = 500)
v1.shape
def cust_cov(vec1, vec2):
    ex_v1 = expected_value(vec1)
    ex_v2 = expected_value(vec2)
    return expected_value((v1-ex_v1)*(v2-ex_v2))

cust_cov(v1, v2)
cust_var(v1)
cust_var(v2)
np.cov(np.array([v1,v2]))

## CORRELATION COEFF

def cust_corr(vec1, vec2):
    return cust_cov(vec1, vec2)/(cust_std(v1)*cust_std(v1))

cust_corr(v1,v2)
np.corrcoef(v1,v2)


## ECDF

def cust_ecdf(vec):
    n = len(vec)
    idx = list(range(1,n+1))
    vec_sorted = np.array(sorted(vec))
    df = pd.DataFrame({'idx': idx,'vec':vec_sorted})
    df['prob'] = df['idx']/n
    cust_plot = sns.scatterplot(x = 'vec', y= 'prob', data = df)
    return cust_plot

fig = cust_ecdf(v)
plt.show()


## WYKRES KWANTYLOWY
import pandas as pd
from scipy import stats
import seaborn as sns


def qqplot(vec):
    n = len(vec)
    idx = list(range(1,n+1))
    vec_sorted = np.array(sorted(vec))
    df = pd.DataFrame({'idx': idx, 'vec':(vec_sorted-vec_sorted.mean())/vec_sorted.std()})
    df['z'] = stats.norm.ppf(df['idx']/n) 
    cust_plot = sns.scatterplot(x = 'vec', y= 'z', data = df)
    return cust_plot

fig = qqplot(v)
plt.show()

## MOMENTS
def cust_centr_moment(vec,p):
    return (((vec-vec.mean())**p).sum()/len(vec))

def cust_moment(vec,p):
    return (vec**p).sum()/len(vec)


## SKEWNESS
def cust_skewness(vec):
    return cust_centr_moment(vec,3)/cust_std(vec)**3

## KURTOSIS
def cust_kurtosis(vec):
    return cust_centr_moment(vec,4)/cust_std(vec)**4
