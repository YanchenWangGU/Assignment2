#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:15:39 2017

@author: jun
"""
import pandas as pd
import numpy as np
import sys

def statistic(df, attribute):
    columnlist=df[attribute].tolist()
    mean = np.mean(columnlist)
    median = np.median(columnlist)
    sd = np.std(columnlist)
    print('The attribute', attribute, 'has a mean of',mean,', median of',median,', standard deviation of', sd)

def main(argv):
    crashdf = pd.read_csv('crash_afterPart1.csv' , sep=',', encoding='latin1')
    carlist= ['BICYCLISTSIMPAIRED','DRIVERSIMPAIRED','FATAL_BICYCLIST',
            'FATAL_DRIVER','FATAL_PEDESTRIAN','MAJORINJURIES_BICYCLIST',
            'MAJORINJURIES_DRIVER','MAJORINJURIES_PEDESTRIAN',
            'MINORINJURIES_BICYCLIST','MINORINJURIES_DRIVER',
            'MINORINJURIES_PEDESTRIAN','PEDESTRIANSIMPAIRED',
            'TOTAL_BICYCLES','TOTAL_GOVERNMENT','TOTAL_PEDESTRIANS',
            'TOTAL_TAXIS','TOTAL_VEHICLES','Total_injuries','Total_involved']
    for i in carlist:
        statistic(crashdf, i)
    
    weatherdf = pd.read_csv('weather_afterPart1.csv' , sep=',', encoding='latin1')
    weatherlist=['PRCP','SNOW','TMAX','TMIN']
    for i in weatherlist:
        statistic(weatherdf, i)
        
if __name__ == "__main__":
    main(sys.argv)