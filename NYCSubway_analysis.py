# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 10:24:24 2014

@author: user
"""
'''Project Overview
In this project, you look at the NYC Subway data and figure out if more people ride the subway when it is raining versus when it is not raining.
You will wrangle the NYC subway data, use statistical methods and data visualization to draw an interesting conclusion about the subway dataset that you've analyzed.
'''
import pandas
import numpy
import scipy.stats
import os
from ggplot import *

df = pandas.read_csv("turnstile_weather_v2.csv")

os.getcwd()
os.listdir(".") #list files
os.system('notepad') #open application

dir(df)
df.describe()
len(df.index) #rows
''' Section 1. Statistical Test

1.	Which statistical test did you use to analyse the NYC subway data? Did you use a one-tail or a two-tail P value?

The 2 sided - Welch T-test as we are comparing the means of two groups. Ridership on rainy days and ridership on non-rainy days. I used a two-tail P value as the question we are
answering is looking for any significant difference whether positive or negative (ridership could be significantly higher or significantly lower on rainy days compared to non-rainy days) even though common intuition might says rainy days should have higher ridership (as people don't like walking in the rain).

2.	Why is this statistical test applicable to the dataset?

The variances of two distributions is unequal (they have unequal stds therefore unequal variances as well since variance = std^2). This test is used to check if two populations have equal means (exactly what we are trying to find in our question).  

3.	What results did you get from this statistical test? These should include the following numerical values: p-values, as well as the means for each of the two samples under test.

rain group mean = 2028
norain group mean = 1845
difference between group means = 182
p-value = 4.64 e-07 

4.	What is the significance and interpretation of these results?

So it looks like on average, more people ride the subway on rainy days vs non-rainy days. But is the difference statistically significant:
p-value = 4.64 e-07 is less than alpha = 0.05 (5% significance level), which means we can reject the null hypothesis that there is no difference between the rainy day and non-rainy day groups.

'''
#subset of raining
raindf = df[df.rain != 0]
raindf = raindf.reset_index(drop=True) #needed for ggplot to plot graph
len(raindf.index) #rows
#subset of not_raining
noraindf = df[df.rain == 0]
noraindf = noraindf.reset_index(drop=True) #needed for ggplot to plot graph
len(noraindf.index) #rows
#compare the means of ridership the two subsets - see if there is a significant difference

len(df.index)==len(raindf.index)+len(noraindf.index) #check that it got subsetted properly, no NAs

#Lets find the ridership variable.
names = list(df.columns.values); names[6]

#Find the average ENTRIESn_hourly - is this actually hourly or do you have to divide by the hours?

raindf.ENTRIESn_hourly.mean(); noraindf.ENTRIESn_hourly.mean()
raindf.ENTRIESn_hourly.describe(); noraindf.ENTRIESn_hourly.describe()
#different sds therefore different varianaces therefore Welch T-test
difference = raindf.ENTRIESn_hourly.mean() - noraindf.ENTRIESn_hourly.mean(); difference
#is this difference statistically significant?

#Ho - Null Hypothesis - muRain - muNoRain = 0 - There is no significant difference in ridership on rainy vs non-rainy days
#Ha - Alternative Hypothesis - muRain - muNoRain != 0 - There is a significant difference in ridership on rainy vs non-rainy days
#Two-tailed test

#Comparing 2 means 
#2 sided - Welch T-test - unequal variances 

tupttest = scipy.stats.ttest_ind(raindf.ENTRIESn_hourly,noraindf.ENTRIESn_hourly,equal_var = False)
tupttest
if tupttest[1] < 0.05: #95% significance level
    result = (False, tupttest)
else:
    result = (True, tupttest)
result
#There is a difference at the 5% significance level as the p-value is less than

%reset #remove all variables from the variable explorer

'''
Section 2. Linear Regression
1.	What approach did you use to compute the coefficients theta and produce prediction for ENTRIESn_hourly in your regression model:
    1.	Gradient descent (as implemented in exercise 3.5)
    2.	OLS using Statsmodels
    3.	Or something different?
    
I used the OLS (Ordinary Least Squares) - Multiple Linear Regression using Statsmodels.

2.	What features (input variables) did you use in your model? Did you use any dummy variables as part of your features?

3.	Why did you select these features in your model? We are looking for specific reasons that lead you to believe that
the selected features will contribute to the predictive power of your model. Your reasons might be based on intuition. For example,  response for fog might be: “I decided to use fog because I thought that when it is very foggy outside people might decide to use the subway more often.” Your reasons might also be based on data exploration and experimentation, for example: “I used feature X because as soon as I included it in my model, it drastically improved my R2 value.”  

4.	What is your model’s R2 (coefficients of determination) value?

5.	What does this R2 value mean for the goodness of fit for your regression model? 

Do you think this linear model to predict ridership is appropriate for this dataset, given this R2  value?
'''

''' 
y = ENTRIESn_hourly
Find out what dependent variables to pick in the model (x).
Get summary statistics.
'''

