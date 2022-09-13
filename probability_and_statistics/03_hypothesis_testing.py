#%% TESTS FOR MEAN  ################
####################################


from scipy import stats
import numpy as np

###############
# SINGLE SAMPLE
x = np.random.randn(100)

stats.ttest_1samp(x, popmean = 0)
stats.ranksums(x, x)

#########################
# TWO INDEPENDENT SAMPLES
x = np.random.randn(100)
y = np.random.randn(100)


stats.ttest_ind(x,y, axis=0, equal_var=True, nan_policy='propagate')
stats.wilcoxon(x, y, zero_method='wilcox', correction=False, alternative='two-sided')
scipy.stats.mannwhitneyu(x, y)
####################
# TWO PAIRED SAMPLES
stats.ttest_rel(x, y, axis = 0,  nan_policy='propagate')

############
# >2 SAMPLES - ANOVA

x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)


stats.f_oneway(x,y,z)

#%%  TESTS FOR EQUALITY OF VARIANCE

def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    f = np.var(x, ddof=1)/np.var(y, ddof=1) #calculate F test statistic 
    dfn = x.size-1 #define degrees of freedom numerator 
    dfd = y.size-1 #define degrees of freedom denominator 
    p = 1-stats.f.cdf(f, dfn, dfd) #find p-value of F test statistic 
    return f, p

#perform F-test
f, p = f_test(x, y)
p


stats.levene(x,y,z)
stats.fligner(x,y,z)
stats.bartlett(x,y,z)


#%% NORMALITY TESTS
x = np.random.randn(100)

stats.shapiro(x)
stats.normaltest(x) #dAgostino test
stats.anderson(x)
stats.kstest(x, 'norm') #kołmogorow-smirnov


from statsmodels.graphics.gofplots import qqplot
qqplot(x)

#%% EQUAL DISTRIBUTIONS
x = np.random.randn(100)
y = np.random.randn(100)
stats.ks_2samp(x, y)

#%% CORRELATION TESTS

stats.pearsonr(x, y)
stats.spearmanr(x, y)
stats.kendalltau(x, y)


#%% TESTS FOR PROPORTIONS

from statsmodels.stats.proportion import proportions_ztest
count = np.array([5, 12])
nobs = np.array([83, 99])
stat, pval = proportions_ztest(count, nobs)
print('{0:0.3f}'.format(pval))


#%% TESTS FOR INDEPENDENCE

# Categorical variables

table = [	[10, 20, 30],
			[6,  9,  17]]
stat, p, dof, expected = stats.chi2_contingency(table)
p




#%% TESTS FOR 

x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)


stats.mannwhitneyu(x,y)
stats.wilcoxon(x,y)
stats.kruskal(x,y)
stats.friedmanchisquare(x,y,z)

#%% skewness

stats.skewtest(x)

#%% kurtosis

stats.kurtosistest(x)

