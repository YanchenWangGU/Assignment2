#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 20:54:03 2017

@author: jun
"""
import pandas as pd
import numpy as np
import sys

def modifydf(df):
    df = df.drop('UNKNOWNINJURIES_BICYCLIST', 1)
    df = df.drop('UNKNOWNINJURIES_DRIVER', 1)
    df = df.drop('UNKNOWNINJURIES_PEDESTRIAN', 1)
    
    df['Total_injuries'] = df['MAJORINJURIES_BICYCLIST'] + df['MAJORINJURIES_DRIVER']+\
    df['MAJORINJURIES_PEDESTRIAN']+ df['MINORINJURIES_BICYCLIST']+ df['MINORINJURIES_DRIVER']+\
    df['MINORINJURIES_PEDESTRIAN']+df['FATAL_BICYCLIST']+ df['FATAL_DRIVER']+ df['FATAL_PEDESTRIAN']
    
    df['Total_involved'] = df['TOTAL_BICYCLES']+df['TOTAL_PEDESTRIANS'] + df['TOTAL_VEHICLES']
    return(df)
    
def getModifiedIQR(df,attribute):
    modifyIQR=np.percentile(df[attribute],95)-np.percentile(df[attribute],5)
    print('Modified IQR for attribute',attribute,'is', modifyIQR)
    
def removeoutlier(df, attribute, outlierdf):
    columnlist=df[attribute].tolist()
    modifyIQR=np.percentile(columnlist,95)-np.percentile(columnlist,5)
    lowerlim=np.percentile(columnlist,5)-1.5*modifyIQR
    upperlim=np.percentile(columnlist,95)+1.5*modifyIQR
    outlierdf=pd.concat([df[(df[attribute]<lowerlim) | (df[attribute]>upperlim)],outlierdf])
    df=df[(lowerlim <= df[attribute]) & (df[attribute]<= upperlim)]
    return([df, outlierdf])
    
def main(argv):
    df = pd.read_csv('crash_beforePart1.csv' , sep=',', encoding='latin1')
    df = modifydf(df)
    cleandf=pd.DataFrame()
    cleandf=df
    outlierdf=pd.DataFrame()
    checkset1 = ['BICYCLISTSIMPAIRED','DRIVERSIMPAIRED','FATAL_BICYCLIST',
            'FATAL_DRIVER','FATAL_PEDESTRIAN','MAJORINJURIES_BICYCLIST',
            'MAJORINJURIES_DRIVER','MAJORINJURIES_PEDESTRIAN',
            'MINORINJURIES_BICYCLIST','MINORINJURIES_DRIVER',
            'MINORINJURIES_PEDESTRIAN','PEDESTRIANSIMPAIRED',
            'TOTAL_BICYCLES','TOTAL_GOVERNMENT','TOTAL_PEDESTRIANS',
            'TOTAL_TAXIS','TOTAL_VEHICLES']
    for i in checkset1:
        getModifiedIQR(df,i)
    
    # We can see that most modified IQR is 0 this is reasonable beacuse most values
    # in some of the attributes are 0 such as injuries of bicylist
    # So we detect outliers by aggregating the injuries and total vechicles involved

    checkset2 = ['Total_injuries', 'Total_involved']
    for i in checkset2:
        [cleandf,outlierdf] = removeoutlier(df, i, outlierdf)
    
    cleandf=cleandf[~cleandf.duplicated(['BICYCLISTSIMPAIRED','DRIVERSIMPAIRED','FATAL_BICYCLIST',
            'FATAL_DRIVER','FATAL_PEDESTRIAN','MAJORINJURIES_BICYCLIST',
            'MAJORINJURIES_DRIVER','MAJORINJURIES_PEDESTRIAN',
            'MINORINJURIES_BICYCLIST','MINORINJURIES_DRIVER',
            'MINORINJURIES_PEDESTRIAN','PEDESTRIANSIMPAIRED',
            'TOTAL_BICYCLES','TOTAL_GOVERNMENT','TOTAL_PEDESTRIANS',
            'TOTAL_TAXIS','TOTAL_VEHICLES','FROMDATE','LATITUDE','LONGITUDE'])]
    
    # Save the dataset which has no outlier into files 
    cleandf.to_csv('crash_afterPart1.csv',sep = ',', index = False)
    outlierdf.to_csv('crash_outlier.csv',sep = ',', index = False)
    
if __name__ == "__main__":
    main(sys.argv)