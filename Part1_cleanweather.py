# import libraries
import pandas as pd
import numpy as np
import sys

# This funtion would deal with problem with missing data and incorrect data
# It takes a dataframe, an attribute name, and a pre-set difference for checking incorrect values
# It will return the cleaned dataframe
def cleanattribute(df, attribute, difference):
    
    # get all the row that the input of attribute is missing
    NAList = df[df[attribute].isnull()].index.tolist() 
    for i in NAList:
        date = df.iloc[i]['Date']
        # check if there is at least one non missing input data for the pass-in attribute of that day
        if(not all(val is None for val in df[df['Date']==date][attribute])):
            mean = np.mean(df[df['Date']==date][attribute])
            df.loc[i,attribute] = mean
            
    ## correct incorrect data in pass-in attribute
    dateList = pd.unique(df['Date']).tolist()
    for i in range(len(dateList)):
        date = dateList[i]
        # we define incorrect if the maximum - minimum for that day is more than difference
        if((max(df[df['Date']==date][attribute]) - min(df[df['Date']==date][attribute]))>difference):
            mean = np.mean(df[df['Date']==date][attribute])
            # To fix it, we replace the value by taking mean for that day
            df.loc[df['Date']==date,attribute] = mean
    
    #return the cleaned dataframe
    return(df)

# This function would count the incorrect values after clean the data
# It takes a dataframe, an attribute, and a pre-set difference for checking incorrect values
# It will print out the checking results for the pass-in attribute
def aftercleancount(df, attribute, difference):
    count = 0
    
    # sort data by date
    dateList = pd.unique(df['Date']).tolist()
    for i in range(len(dateList)):
        date = dateList[i]
        #check the min-max difference is within the pre-set allowance
        if((max(df[df['Date']==date][attribute]) - min(df[df['Date']==date][attribute]))>difference):
            count = count+1
            
    # print the test results
    print('After cleaning the attribute', attribute,'the number of incorrect value is reduced to', count)
    
    
def main(argv):
    
    # read the weather condition data from project 
    df = pd.read_csv('weather_beforePart1.csv' , sep=',', encoding='latin1')
    
    # clean the TMAX and TMIN attributes(the rest of the attributes were checked 
    # and cleaned at project assginment1)
    cleanset = ['TMAX','TMIN']
    for i in cleanset:
        df = cleanattribute(df,i, 10)
        aftercleancount(df,i, 10)
        
    # aftering calling the function, there are still some missing values
    # so we search the history weather for still missing values and fill in manually
    df.loc[df['Date']=='2014-01-18T00:00:00','SNOW']=0
    df.loc[df['Date']=='2015-01-12T00:00:00','SNOW']=0
    df.loc[df['Date']=='2015-01-23T00:00:00','SNOW']=0
    df.loc[df['Date']=='2016-01-07T00:00:00','SNOW']=0
    df.loc[df['Date']=='2016-01-25T00:00:00','SNOW']=0
    df.loc[df['Date']=='2014-01-18T00:00:00','SNOW']=0
    df.loc[df['Date']=='2015-01-23T00:00:00','SNOW']=0
    df.loc[df['Date']=='2014-05-11T00:00:00','SNOW']=0
    df.loc[df['Date']=='2014-05-11T00:00:00','TMAX']=81
    df.loc[df['Date']=='2014-05-11T00:00:00','TMIN']=62
    df.loc[df['Date']=='2013-08-31T00:00:00','TMAX']=92
    df.loc[df['Date']=='2013-08-31T00:00:00','TMIN']=73
    
    # save the cleaned weather condition data
    df.to_csv('weather_afterPart1.csv',sep = ',', index = False)

if __name__ == "__main__":
    main(sys.argv)
    