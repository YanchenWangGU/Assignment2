# import libraries
import pandas as pd
import numpy as np
import sys

# This function take a dataframe and attribute name as inputs 
# and will print out the statistic results (mean, median and standarddeviation
# for that attribute
def statistic(df, attribute):
    columnlist=df[attribute].tolist()
    # detemine the mean, median, and standard deviation for the passed-in attribute
    mean = np.mean(columnlist)
    median = np.median(columnlist)
    sd = np.std(columnlist)
    # print out the results
    print('The attribute', attribute, 'has a mean of',mean,', median of',
          median,', standard deviation of', sd,'\n')

def main(argv):
    # read the cleaned car-crash data
    crashdf = pd.read_csv('crash_afterPart1.csv' , sep=',', encoding='latin1')
    # do statistic for all the attributs for car-crash data
    carlist= ['BICYCLISTSIMPAIRED','DRIVERSIMPAIRED','FATAL_BICYCLIST',
            'FATAL_DRIVER','FATAL_PEDESTRIAN','MAJORINJURIES_BICYCLIST',
            'MAJORINJURIES_DRIVER','MAJORINJURIES_PEDESTRIAN',
            'MINORINJURIES_BICYCLIST','MINORINJURIES_DRIVER',
            'MINORINJURIES_PEDESTRIAN','PEDESTRIANSIMPAIRED',
            'TOTAL_BICYCLES','TOTAL_GOVERNMENT','TOTAL_PEDESTRIANS',
            'TOTAL_TAXIS','TOTAL_VEHICLES','Total_injuries','Total_involved']
    for i in carlist:
        statistic(crashdf, i)
    
    # read the cleaned weather condition data
    weatherdf = pd.read_csv('weather_afterPart1.csv' , sep=',', encoding='latin1')
    # do statistic for all the attributs for weather condition data
    weatherlist=['PRCP','SNOW','TMAX','TMIN']
    for i in weatherlist:
        statistic(weatherdf, i)
        
if __name__ == "__main__":
    main(sys.argv)