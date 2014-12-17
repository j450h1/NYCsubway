# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 12:40:20 2014

@author: user
"""

#Graphs to show distribution
'''
ggrain = ggplot(aes(x='ENTRIESn_hourly'), data=raindf) + \
    geom_histogram() + \
    ggtitle("Histogram of Hourly Entries (Rain)") + \
    labs("HOURLY Entries (Rain)", "Frequency")    
ggnorain = ggplot(aes(x='ENTRIESn_hourly'), data=noraindf) + \
    geom_histogram() + \
    ggtitle("Histogram of Hourly Entries (No Rain)") + \
    labs("HOURLY Entries (No Rain)", "Frequency")    
'''

#Create a new categorical variable for rain column