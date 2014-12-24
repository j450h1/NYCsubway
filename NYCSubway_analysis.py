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
import numpy as np
import scipy.stats
import os
from ggplot import *

df = pandas.read_csv("turnstile_weather_v2.csv")
df = df.reset_index(drop=True)
df.index = range(df.shape[0])

df.head()

os.getcwd()
os.listdir(".") #list files
os.system('notepad') #open application

dir(df)
df.describe()

df['rain'].describe()
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
percentrain = float(len(raindf.index))/float(len(df)) 
percentrain
#subset of not_raining
noraindf = df[df.rain == 0]
noraindf = noraindf.reset_index(drop=True) #needed for ggplot to plot graph
len(noraindf.index) #rows
percentnorain = float(len(noraindf.index))/float(len(df)) 
percentnorain
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
1.	What approach did you use to compute the coefficients theta and produce 
prediction for ENTRIESn_hourly in your regression model:
    1.	Gradient descent (as implemented in exercise 3.5)
    2.	OLS using Statsmodels
    3.	Or something different?
-----------------------------------------------------------------------------------    
I used the OLS (Ordinary Least Squares) - Multiple Linear Regression using Statsmodels.

2.	What features (input variables) did you use in your model? Did you use 
any dummy variables as part of your features?
--------------------------------------------------------------------------------------
'hour','weekday','rain','fog','tempi'
dummy variables - Station and Unit

3.	Why did you select these features in your model? We are looking for specific reasons 
that lead you to believe that the selected features will contribute to the predictive power 
of your model. Your reasons might be based on intuition. For example,  response for fog 
might be: “I decided to use fog because I thought that when it is very foggy outside people 
might decide to use the subway more often.” Your reasons might also be based on data exploration 
and experimentation, for example: “I used feature X because as soon as I included it in my model,
it drastically improved my R2 value.”  
-------------------------------------------------------------------------------------------------
I used hour because it was highly linearly correlated with y (ENTRIESn_hourly) relative to the other features
and intuitevely because ridership would likely vary based on the time of day as more
people would probably ride the subway to and from work from 7 to 9 am and 4 to 6 pm
I chose weekday because I would expect ridership to be higher on the weekdays when
most people are commuting to work as opposed to the weekend when they are more likely to be at home.
I chose station and units as dummy variables because larger/more trafficked stations and units would be expected to have more riders as opposed to smaller stations or units.
I initally chose rain and fog because it is expected that people would ride the subway more during bad weather instead of walking outside.
Same with tempi it would be expected that people prefer to walk outside during the summer when the temperature is higher. 
However, after computing R2 I found these three variables did not help increase the predictive power of the model as R2 did not improve when adding them to the model.

4.	What is your model’s R2 (coefficients of determination) value?
-------------------------------------------------------------------------------------------------

0.63

5.	What does this R2 value mean for the goodness of fit for your regression model? 
-------------------------------------------------------------------------------------------------
 63% of the variability in the dependent variable, ENTRISn_hourly, is explained
 by this model which has 5 continious variable and 2 categorical variables which have been converted to hundreds of dummy variables. This means that it is a relatively good fit for the model since
 the values of R2 can vary from 0 to 1. 
Do you think this linear model to predict ridership is appropriate for this dataset, given this R2  value?
-----------------------------------------------------------------------------------------------------------
Yes, I think it is the appropriate model for this data set as the relationship definately looks linear. In fact, if 
you take the square root of r-squared you get r, the correlation coefficient, which is + or - 0.79 which indicates a very strong linear relationship.
'''

'''
y = ENTRIESn_hourly
Find out what dependent variables to pick in the model (x).
Get summary statistics.
'''
#Let's see the correlation between variables.

#We cannot use EXITSn_hourly as a predictor
#Let's look at the correlations

#What to do with N/A values?
stationDUMMY = pandas.get_dummies(df['station'],prefix='station')
UNITDUMMY = pandas.get_dummies(df['UNIT'],prefix='unit')

y = df['ENTRIESn_hourly']
#x will be chosen features/regressors
x = df.copy()
df = df.join(stationDUMMY)
df = df.join(UNITDUMMY)

#new

#x = df.ix[:,'b':] - if doing a range of columns
#possible features
#Do we need to subset into a new df? Yes, easier to read and understand
columns = ['hour','weekday','rain','fog','tempi']
columns = ['hour','weekday'] #can get 0.63 just from these two variables
#adding more variables doesn't increase r-squared
 #no rain in model #seems to be same r-squared
#weekday seems to increase the r-squared value the most

numerical_columns = ['hour','day_week','weekday','latitude','longitude','fog','precipi','pressurei','rain','tempi','wspdi','meanprecipi','meanpressurei','meantempi','meanwspdi', 'weather_lat', 'weather_lon']
#x = df[numerical_columns]
x = df[columns]
x = x.join(stationDUMMY)
x = x.join(UNITDUMMY)
#x = x.drop('weekday',1)
x.columns
x
x.tail()
y
len(x)
numpy.corrcoef(y,df['EXITSn_hourly']) #r = 0.64
numpy.corrcoef(y,df['ENTRIESn_hourly']) #r = 1
numpy.corrcoef(y,df['UNIT']) #error
numpy.corrcoef(y,df['DATEn']) #error
numpy.corrcoef(y,df['TIME']) #error
numpy.corrcoef(y,df['datetime']) #error
numpy.corrcoef(y,df['station']) #error #used dummies
numpy.corrcoef(y,df['conds']) #error 
numpy.corrcoef(y,df['hour']) #0.28
numpy.corrcoef(y,df['day_week']) #0.09
numpy.corrcoef(y,df['weekday']) #0.14
numpy.corrcoef(y,df['latitude']) #0.11
numpy.corrcoef(y,df['longitude']) #-0.12
numpy.corrcoef(y,df['fog']) #-0.008
numpy.corrcoef(y,df['precipi']) #-0.027
numpy.corrcoef(y,df['pressurei']) #-0.03
numpy.corrcoef(y,df['rain']) #0.025
numpy.corrcoef(y,df['tempi']) #0.089 
numpy.corrcoef(y,df['wspdi']) #-0.056
numpy.corrcoef(y,df['meanprecipi']) #-0.035 #check if precipi is greater than average to create new feature
numpy.corrcoef(y,df['meanpressurei']) #-0.006 #check if pressure is greater than average to create new feature
numpy.corrcoef(y,df['meantempi']) #-0.026 #check if greater than average to create new feature
numpy.corrcoef(y,df['meanwspdi']) #-0.0039 #check if greater than average to create new feature
numpy.corrcoef(y,df['weather_lat']) #0.089 
numpy.corrcoef(y,df['weather_lon']) #-0.137

import math 
math.sqrt(0.63)

import statsmodels.api as sm
model = sm.OLS(y,x)
results = model.fit()
results.rsquared #0.63


results.params
results.tvalues
results.pvalues
results.rsquared_adj

predictions = results.predict(x)
#residuals
plt.figure()
(y - predictions).hist()
dir(results)
'''
Section 3. Visualization
Please include two visualizations that show the relationships between two or 
more variables in the NYC subway data. You should feel free to implement something
that we discussed in class (e.g., scatter plots, line plots, or histograms) or 
attempt to implement something more advanced if you'd like.
Remember to add appropriate titles and axes labels to your
plots. Also, please add a short description below each figure commenting on the 
key insights depicted in the figure. 
1.	One visualization should contain two histograms: one of  ENTRIESn_hourly for
rainy days and one of ENTRIESn_hourly for non-rainy days. You can combine the two
histograms in a single plot or you can use two different plots.
For the histogram, you should have intervals representing the volume of ridership
(value of ENTRIESn_hourly) on the x-axis and the frequency of occurrence on the y-axis. For example, you might have one interval (along the x-axis) with values from 0 to 1000. The height of the bar for this interval will then represent the number of records (rows in our data) that have ENTRIESn_hourly that fall into this interval.
Remember to increase the number of bins in the histogram (by having larger number of bars). The default bin width is not sufficient to capture the variability in the two samples.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
2.	One visualization can be more freeform, some suggestions are:
1.	Ridership by time-of-day or day-of-week
2.	How ridership varies by subway station
3.	Which stations have more exits or entries at different times of day
'''
ggrain = ggplot(aes(x='ENTRIESn_hourly'), data=raindf) + \
    geom_histogram() + \
    ggtitle("Histogram of Hourly Entries (Rain)") + \
    labs("HOURLY Entries (Rain)", "Frequency")    
ggrain
ggnorain = ggplot(aes(x='ENTRIESn_hourly'), data=noraindf) + \
    geom_histogram() + \
    ggtitle("Histogram of Hourly Entries (No Rain)") + \
    labs("HOURLY Entries (No Rain)", "Frequency")    
ggnorain
#We need to plot on the same chart

df = df[['ENTRIESn_hourly','rain']]
df = df.reset_index(drop=True)
df.index = range(df.shape[0])
histogram = ggplot(aes(x='ENTRIESn_hourly'), data=df) + \
    geom_histogram(data = df[df['rain'] == 0], fill = "yellow", binwidth = 300) + \
    geom_histogram(data = df[df['rain'] != 0], fill = "blue", binwidth = 300) + \
    xlim(low=0, high=7000) + \    
    ggtitle("Histogram of Hourly Entries to the NYC Subway") + \
    labs("HOURLY Entries (Yellow - No Rain, Blue - Rain)", "Frequency")    
#From the histogram, it appears the shape of the distribution for rain any non-rainy days
is near identical. The differences in height can be attributed to the fact that the majority of the data
(over 3/4) was collected on non-rainy days. If we had an equal amount of data from rainy days and non-rainy days 
then the height difference would matter more. From the chart, it doesn't appear there is any major difference, but these
are total counts and doesn't show the entire picture.

second visualization done in tableau

3. From three different methods, we are able to confirm that ridership of the NYC Subway
is indeed higher on non-rainy days vs rainy days. Average ridership is a better measurement to determine 
if rain vs no-rain makes a difference.:
    
    Although the average difference is statistically significant, for all practical purposes
    there is no observable difference in ridership on non-rainy vs rainy days. This also makes sense
    intuitively as most of the time people take the subway to save commuting time and money. It wouldn't make
    much sense to take a cab or walk just because it is a sunny day. Most people would not change their commuting
    habits due to the weather(except snow or extreme weather for example). It would be interesting to see what kind of
    data is available in winter months. Also many people don't even check the weather forecast before leaving or
    planning their commute so they have already planned that they will be taking the subway.