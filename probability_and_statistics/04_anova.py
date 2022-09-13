import pandas as pd
from scipy import stats

data = pd.read_csv('C:/Users/Julita/Documents/LEARN/DATA_SCIENCE/R/RBook/datasets/twosample.csv')

df_aov1 = data[['a', 'x']]
df_aov2 = data[['a','b', 'x']]


# ASSUMPTIONS

#Residuals (experimental error) are normally distributed (Shapiro-Wilks Test)
#Homogeneity of variances (variances are equal between treatment groups) (Levene’s or Bartlett’s Test)
#Observations are sampled independently from each other

import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.boxplot(x='a', y='x', data=df_aov1, color='#99c2a2')
ax = sns.swarmplot(x="a", y="x", data=df_aov1, color='#7d0013')


############
#%% ONE WAY ANOVA


s, p = stats.f_oneway(df_aov1.loc[df_aov1['a'] =='one','x'],
                      df_aov1.loc[df_aov1['a'] =='two', 'x'],
                      df_aov1.loc[df_aov1['a'] =='three', 'x'],   
                      df_aov1.loc[df_aov1['a'] =='four', 'x'],
                      df_aov1.loc[df_aov1['a'] =='five', 'x']                          
                      )

s
p

import statsmodels.api as sm
from statsmodels.formula.api import ols

# Ordinary Least Squares (OLS) model
model = ols('x ~ C(a)', data=df_aov1).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table


# ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=df_aov1, res_var='x', anova_model='x ~ C(a)')
res.anova_summary

#####################################
# Tests post-hoc (Tukey HSD)

from bioinfokit.analys import stat
# perform multiple pairwise comparison (Tukey's HSD)
# unequal sample size data, tukey_hsd uses Tukey-Kramer test
res = stat()
res.tukey_hsd(df=df_aov1, res_var='x', xfac_var='a', anova_model='x ~ C(a)')
res.tukey_summary

'''
## TukeyHSD
import statsmodels.stats.multicomp as mc

comp = mc.MultiComparison(df_aov1['a'],  )
post_hoc_res = comp.tukeyhsd()
post_hoc_res.summary()
'''

#####################################
### NORMALITY
# QQ-plot
import statsmodels.api as sm
import matplotlib.pyplot as plt
# res.anova_std_residuals are standardized residuals obtained from ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()


w, pvalue = stats.shapiro(model.resid)
print(w, pvalue)

##################################
# Check the homogeneity of variance assumption

# load packages
import scipy.stats as stats
w, pvalue = stats.bartlett(df_aov1.loc[df_aov1['a'] =='one','x'],
                      df_aov1.loc[df_aov1['a'] =='two', 'x'],
                      df_aov1.loc[df_aov1['a'] =='three', 'x'],   
                      df_aov1.loc[df_aov1['a'] =='four', 'x'],
                      df_aov1.loc[df_aov1['a'] =='five', 'x'] )
print(w, pvalue)


# if you have a stacked table, you can use bioinfokit v1.0.3 or later for the bartlett's test
from bioinfokit.analys import stat 
res = stat()
res.bartlett(df=df_aov1, res_var='x', xfac_var='a')
res.bartlett_summary



# if you have a stacked table, you can use bioinfokit v1.0.3 or later for the Levene's test
from bioinfokit.analys import stat 
res = stat()
res.levene(df=df_aov1, res_var='x', xfac_var='a')
res.levene_summary



#%%TWO WAY ANOVA

import statsmodels.api as sm
from statsmodels.formula.api import ols
# Ordinary Least Squares (OLS) model
# C(Genotype):C(years) represent interaction term
model = ols('x ~ C(a) + C(b) + C(a):C(b)', data=df_aov2).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table

# ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
from bioinfokit.analys import stat
res = stat()
res.anova_stat(df=df_aov2, res_var='value', anova_model='x ~ C(a) + C(b) + C(a):C(b)')
res.anova_summary


##########################
# INTERACTION PLOT

from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
fig = interaction_plot(x=df_aov2['a'], trace=df_aov2['b'], response=df_aov2['x'])
plt.show()

###########################
# Tukey HSD
from bioinfokit.analys import stat
# perform multiple pairwise comparison (Tukey HSD)
# unequal sample size data, tukey_hsd uses Tukey-Kramer test
res = stat()
# for main effect Genotype
res.tukey_hsd(df=df_aov2, res_var='x', xfac_var=['a','b'], anova_model='x ~ C(a) + C(b) + C(a):C(b)')
res.tukey_summary

res.tukey_hsd(df=df_aov2, res_var='x', xfac_var='a', anova_model='x ~ C(a) + C(b) + C(a):C(b)')
res.tukey_summary

#########################
# NORMALITY

# QQ-plot
import statsmodels.api as sm
import matplotlib.pyplot as plt
# res.anova_std_residuals are standardized residuals obtained from two-way ANOVA (check above)
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()

# Shapiro-Wilk test
import scipy.stats as stats
w, pvalue = stats.shapiro(res.anova_model_out.resid)
print(w, pvalue)

###############################
# Check the homogeneity of variance assumption

from bioinfokit.analys import stat 
res = stat()
res.levene(df=df_aov2, res_var='x', xfac_var=['a', 'b'])
res.levene_summary
