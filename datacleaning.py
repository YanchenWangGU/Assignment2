#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:59:51 2017

@author: jun
"""

import pandas as pd
import numpy as np
import sys

def cleanattribute(dfNew, attribute, difference):
    NAList = dfNew[dfNew[attribute].isnull()].index.tolist()
    
    # print(dfNew['TMAX'].isnull().values.ravel().sum())
    # 1173
    for i in NAList:
        date = dfNew.iloc[i]['Date']
        # check if there is at least one non missing TMAX data for that day
        if(not all(val is None for val in dfNew[dfNew['Date']==date][attribute])):
            mean = np.mean(dfNew[dfNew['Date']==date][attribute])
            dfNew.loc[i,attribute] = mean
            
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
        if((max(dfNew[dfNew['Date']==date][attribute]) - min(dfNew[dfNew['Date']==date][attribute]))>difference):
            mean = np.mean(dfNew[dfNew['Date']==date][attribute])
            dfNew.loc[dfNew['Date']==date,attribute] = mean
    
    return(dfNew)

def aftercleancount(dfNew, attribute, difference):
    count = 0
    dateList = pd.unique(dfNew['Date']).tolist()
    for i in range(len(dateList)):
        date = dateList[i]
        if((max(dfNew[dfNew['Date']==date][attribute]) - min(dfNew[dfNew['Date']==date][attribute]))>difference):
            count = count+1
    print('After cleaning the attribute', attribute,'the number of incorrect value is reduced to', count)

def main(argv):
    dfNew = pd.read_csv('weather_beforePart1.csv' , sep=',', encoding='latin1')
    cleanSet = ['TMAX','TMIN']
    for i in cleanSet:
        dfNew = cleanattribute(dfNew,i, 10)
        aftercleancount(dfNew,i, 10)
    dfNew.to_csv('weather_afterPart1.csv',sep = ',', index = False)

if __name__ == "__main__":
    main(sys.argv)
    