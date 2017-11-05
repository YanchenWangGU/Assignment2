# import libraries
import pandas as pd
import numpy as np
import sys

# This function take a dataframe as input, it will modify the existed car-crash data
# This part will create two attributes 'Total_involved' and 'Total_involved' 
# where 'Total_involved' is sum of all injuries and fatalities in the accident
# We are going to use this attribute to evaluate the severity of the accident
# in the later parts 
def modifydf(df):
    
    # drop the three unknown attributes, since these are unrelated to our study
    # and all values in the three attributes are zero
    df = df.drop('UNKNOWNINJURIES_BICYCLIST', 1)
    df = df.drop('UNKNOWNINJURIES_DRIVER', 1)
    df = df.drop('UNKNOWNINJURIES_PEDESTRIAN', 1)
    
    # sum the total number of injuries for futuer analysis
    df['Total_injuries'] = df['MAJORINJURIES_BICYCLIST'] + df['MAJORINJURIES_DRIVER']+\
    df['MAJORINJURIES_PEDESTRIAN']+ df['MINORINJURIES_BICYCLIST']+ df['MINORINJURIES_DRIVER']+\
    df['MINORINJURIES_PEDESTRIAN']+df['FATAL_BICYCLIST']+ df['FATAL_DRIVER']+ df['FATAL_PEDESTRIAN']
    
    # sum the total number of car, bicycles, vehicles, and pedestrians involved for future analysis
    df['Total_involved'] = df['TOTAL_BICYCLES']+df['TOTAL_PEDESTRIANS'] + df['TOTAL_VEHICLES']
    
    #return the modified dataframe
    return(df)
    
# This function takes a dataframe and an attribute name as input
# and it will compute and print out the modified IQR value
def getModifiedIQR(df,attribute):
    # since all the inputs for each attribute are one digite number
    # and a significant number of those are 0, so we decide to modify the
    # IQR value to the difference between the 95 and 5 percentiles
    # (replacing the 75 and 25 percentiles)
    modifyIQR=np.percentile(df[attribute],95)-np.percentile(df[attribute],5)
    print('Modified IQR for attribute',attribute,'is', modifyIQR,'\n')
    

# This function takes a original dataframe, an attribute name, 
# and a outlier dataframe as input. It will use the modified version
# of IQR to check the outliers and move the outliers from the original dataframe
# into the outlier dataframe, then return both updated dataframes.
def removeoutlier(df, attribute, outlierdf):
    columnlist=df[attribute].tolist()
    
    # compute the modified IQR
    modifyIQR=np.percentile(columnlist,95)-np.percentile(columnlist,5)
    # compute the lower and upper bound
    lowerlim=np.percentile(columnlist,5)-1.5*modifyIQR
    upperlim=np.percentile(columnlist,95)+1.5*modifyIQR
    # update the outlier dataframe
    outlierdf=pd.concat([df[(df[attribute]<lowerlim) | (df[attribute]>upperlim)],outlierdf])
    # remove the outlier from the original dataframe
    df=df[(lowerlim <= df[attribute]) & (df[attribute]<= upperlim)]
    
    # return both dataframes
    return([df, outlierdf])
    
def main(argv):
    
    # read the car-crash data from the results of porject assginment 1 
    df = pd.read_csv('crash_beforePart1.csv' , sep=',', encoding='latin1')
    
    # modify the dataframe
    df = modifydf(df)
    cleandf=pd.DataFrame()
    cleandf=df
    outlierdf=pd.DataFrame()
    
    #check the value of modified IQR for all the original attributes in car-carsh data
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
    
    # checking for the duplicated data and remove the duplicated rows
    cleandf=cleandf[~cleandf.duplicated(['BICYCLISTSIMPAIRED','DRIVERSIMPAIRED','FATAL_BICYCLIST',
            'FATAL_DRIVER','FATAL_PEDESTRIAN','MAJORINJURIES_BICYCLIST',
            'MAJORINJURIES_DRIVER','MAJORINJURIES_PEDESTRIAN',
            'MINORINJURIES_BICYCLIST','MINORINJURIES_DRIVER',
            'MINORINJURIES_PEDESTRIAN','PEDESTRIANSIMPAIRED',
            'TOTAL_BICYCLES','TOTAL_GOVERNMENT','TOTAL_PEDESTRIANS',
            'TOTAL_TAXIS','TOTAL_VEHICLES','FROMDATE','LATITUDE','LONGITUDE'])]
    
    # Save the dataset which has no outlier into files 
    cleandf.to_csv('crash_afterPart1.csv',sep = ',', index = False)
    # save all the outliers into another files
    outlierdf.to_csv('crash_outlier.csv',sep = ',', index = False)
    
if __name__ == "__main__":
    main(sys.argv)