#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 13:59:51 2017

@author: jun
"""

import pandas as pd
import numpy as np
import sys

def cleanattribute(df, attribute, difference):
    NAList = df[df[attribute].isnull()].index.tolist()
    
    # print(df['TMAX'].isnull().values.ravel().sum())
    # 1173
    for i in NAList:
        date = df.iloc[i]['Date']
        # check if there is at least one non missing TMAX data for that day
        if(not all(val is None for val in df[df['Date']==date][attribute])):
            mean = np.mean(df[df['Date']==date][attribute])
            df.loc[i,attribute] = mean
            
    # print(df['TMAX'].isnull().values.ravel().sum())
    # 2
    # The missing value reduced from about 1173 entries to 2 entries
    
    ## correct incorrect data in TMAX
    dateList = pd.unique(df['Date']).tolist()
    for i in range(len(dateList)):
        date = dateList[i]
        # we define incorrect if the maximum - minimum for that day is more than 10
        # because it is very unlikely to have more than 10 farenheit difference of max temperture
        # To fix it, we replace the value by taking mean for that day
        if((max(df[df['Date']==date][attribute]) - min(df[df['Date']==date][attribute]))>difference):
            mean = np.mean(df[df['Date']==date][attribute])
            df.loc[df['Date']==date,attribute] = mean
    
    return(df)

def aftercleancount(df, attribute, difference):
    count = 0
    dateList = pd.unique(df['Date']).tolist()
    for i in range(len(dateList)):
        date = dateList[i]
        if((max(df[df['Date']==date][attribute]) - min(df[df['Date']==date][attribute]))>difference):
            count = count+1
    print('After cleaning the attribute', attribute,'the number of incorrect value is reduced to', count)
    
    
def main(argv):
    df = pd.read_csv('weather_beforePart1.csv' , sep=',', encoding='latin1')
    cleanset = ['TMAX','TMIN']
    for i in cleanset:
        df = cleanattribute(df,i, 10)
        aftercleancount(df,i, 10)
    
    #search the history weather for still missing values and fill in manually
    df.loc[df['Date']=='2014-01-18T00:00:00','SNOW']=0
    df.loc[df['Date']=='2015-01-12T00:00:00','SNOW']=0
    df.loc[df['Date']=='2015-01-23T00:00:00','SNOW']=0
    df.loc[df['Date']=='2016-01-07T00:00:00','SNOW']=0
    df.loc[df['Date']=='2016-01-25T00:00:00','SNOW']=17
    df.loc[df['Date']=='2014-01-18T00:00:00','SNOW']=0
    df.loc[df['Date']=='2015-01-23T00:00:00','SNOW']=0
    df.loc[df['Date']=='2014-05-11T00:00:00','SNOW']=0
    df.loc[df['Date']=='2014-05-11T00:00:00','TMAX']=81
    df.loc[df['Date']=='2014-05-11T00:00:00','TMIN']=62
    df.loc[df['Date']=='2013-08-31T00:00:00','TMAX']=92
    df.loc[df['Date']=='2013-08-31T00:00:00','TMIN']=73
    df.to_csv('weather_afterPart1.csv',sep = ',', index = False)

if __name__ == "__main__":
    main(sys.argv)
    