#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 19:44:04 2017

@author: jun
"""

# This part cleans two attributes "TMAX" and "TMIN" in weather dataset
import pandas as pd
import numpy as np

## The codes below will fix missing values and noise values in attributes 
## "TMAX" and "TMIN'
dfNew = pd.read_csv('weather_beforePart1.csv' , sep=',', encoding='latin1')
        
## Replace rest of the missing values by taking mean of the Tmax for that day
## Update missing value list
tmaxNAList = dfNew[dfNew['TMAX'].isnull()].index.tolist()

# print(dfNew['TMAX'].isnull().values.ravel().sum())
# 1173
for i in tmaxNAList:
    date = dfNew.iloc[i]['Date']
    # check if there is at least one non missing TMAX data for that day
    if(not all(val is None for val in dfNew[dfNew['Date']==date]['TMAX'])):
        mean = np.mean(dfNew[dfNew['Date']==date]['TMAX'])
        dfNew.loc[i,'TMAX'] = mean
        
# print(dfNew['TMAX'].isnull().values.ravel().sum())
# 2
# The missing value reduced from about 1173 entries to 2 entries

## correct incorrect data in TMAX
dateList = pd.unique(dfNew['Date']).tolist()
for i in range(len(dateList)):
    date = dateList[i]
    # we define incorrect if the maximum - minimum for that day is more than 10
    # because it is very unlikely to have more than 10 farenheit difference of max temperture
    # To fix it, we replace the value by taking mean for that day
    if((max(dfNew[dfNew['Date']==date]['TMAX']) - min(dfNew[dfNew['Date']==date]['TMAX']))>10):
        mean = np.mean(dfNew[dfNew['Date']==date]['TMAX'])
        dfNew.loc[dfNew['Date']==date,'TMAX'] = mean
        
## Rerun cleanliness and see how it improves 
count = 0
for i in range(len(dateList)):
    date = dateList[i]
    if((max(dfNew[dfNew['Date']==date]['TMAX']) - min(dfNew[dfNew['Date']==date]['TMAX']))>10):
        count = count+1
# print(count)
# 0 
# Number of incorrect values reduces to 0
        
# After fixing missing and incorrect values, there are only 2 missing with no incorrect values

## Fix missing values in TMIN 
## Replace missing values using mean of TMIN for that day
tminNAList = dfNew[dfNew['TMIN'].isnull()].index.tolist()
# print(dfNew['TMIN'].isnull().values.ravel().sum())
# 1173
for i in tminNAList:
    date = dfNew.iloc[i]['Date']
    # Same as TMAX, there must be at least one non missing for that day 
    if(not all(val is None for val in dfNew[dfNew['Date']==date]['TMIN'])):
        mean = np.mean(dfNew[dfNew['Date']==date]['TMIN'])
        dfNew.loc[i,'TMIN'] = mean
        
# print(dfNew['TMIN'].isnull().values.ravel().sum())
# 2
# Number of missing values reduces from 1173 to 0
    
# Correct incorrect data in TMIN by taking mean TMIN for that day
dateList = pd.unique(dfNew['Date']).tolist()
for i in range(len(dateList)):
    date = dateList[i]
    # we define incorrect if the maximum - minimum for that day is more than 10
    # because it is very unlikely to have more than 10 farenheit difference of min temperture
    # To fix it, we replace the value by taking mean for that day
    if((max(dfNew[dfNew['Date']==date]['TMIN']) - min(dfNew[dfNew['Date']==date]['TMIN']))>10):
        mean = np.mean(dfNew[dfNew['Date']==date]['TMIN'])
        dfNew.loc[dfNew['Date']==date,'TMIN'] = mean
        
# After cleaning, we check if there is any other incorrect values
count = 0
dateList = pd.unique(dfNew['Date']).tolist()
for i in range(len(dateList)):
    date = dateList[i]
    if((max(dfNew[dfNew['Date']==date]['TMIN']) - min(dfNew[dfNew['Date']==date]['TMIN']))>10):
        count = count+1
# print(count)
# 0
# There is no missing or incorrect values in TMIN

# Save the cleaned dataset into files 
dfNew.to_csv('weather_afterPart1.csv',sep = ',', index = False)
