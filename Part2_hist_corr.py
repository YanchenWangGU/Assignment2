# import libraries
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sys

# This function takes two same time period dataframes, 
# merges two dataframes into one dataframe that each row
# stands for one specific date, and returns the merged dataframe.
def mergetwodata(myData1, myData2):
    #split the data attribute to date and time 
    myData1['Date'], myData1['Time'] = zip(*myData1['FROMDATE'].apply(lambda x: x.split('T')))
    myData2['Date'], myData2['Time'] = zip(*myData2['Date'].apply(lambda x: x.split('T')))
    
    #find the date that rains and the date that does not rain separately. 
    dateArray = myData2.Date.unique()
    meanRain = []
    meanSnow = []
    accidentCount = []
    injuryCount = []
    maxTemp = []
    minTemp = []
    for i in range(len(dateArray)):
        # count the total number of car accidents and total number of injurys
        accidentCount.append(myData1[myData1.Date == dateArray[i]].shape[0])
        injuryCount.append(np.sum(myData1[myData1.Date == dateArray[i]]['Total_injuries']))
        
        # merge all the weather attributes for the three weather stations by taking the mean
        meanRain.append(np.mean(myData2[myData2.Date == dateArray[i]]['PRCP']))
        meanSnow.append(np.mean(myData2[myData2.Date == dateArray[i]]['SNOW']))
        maxTemp.append(np.mean(myData2[myData2.Date == dateArray[i]]['TMAX']))
        minTemp.append(np.mean(myData2[myData2.Date == dateArray[i]]['TMIN']))
        
    Data = pd.DataFrame()
    #Date is the attribute showing date
    Data['Date'] = dateArray  
    #AccidentCount is the number of accident in the specific date
    Data['AccidentCount'] = accidentCount
    #Injuries is the number of total injuries in the specific data
    Data['Injuries'] = injuryCount
    #MeanSnow is the mean of the snow fall for three different location
    Data['MeanSnow'] = meanSnow
    #MeanRain is the mean of the rain fall for three different location                 
    Data['MeanRain'] = meanRain
    #meanMaxT is the mean of the rain fall for three different location
    Data['meanMaxT'] = maxTemp
    #meanMinT is the mean of the rain fall for three different location
    Data['meanMinT'] = minTemp
    
    #return the merged dataframe
    return(Data)
    


def main(argv):
    #read the cleaned car-crash and cleaned weather condition data
    crashdf = pd.read_csv('crash_afterPart1.csv' , sep=',', encoding='latin1')
    weatherdf = pd.read_csv('weather_afterPart1.csv' , sep=',', encoding='latin1')
    
    #merge the car-crash data and weather condition data 
    mergedf = mergetwodata(crashdf, weatherdf)
    
    # Look the sorted data and detect outliers by hand
    #mergedf.sort_values('AccidentCount')
    #clean the outliers for the after-merge dataframe
    mergedf = mergedf[(mergedf['AccidentCount'] >10) & (mergedf['AccidentCount'] <200)]
    mergedf = mergedf.reset_index(drop=True)
    
 

    # plot scatter plot matrix for the merged data set
    # which contains the attributs of average snowfall, average rainfall, 
    # average TMIN, average TMAX, total number of car accidents per day, 
    # and total number of injurys per day
    scatter_matrix(mergedf, figsize=(15,15))
    #plot the histogram for each attribute in merged dataframe
    mergedf.hist(figsize=(10,10))
    plt.show()

if __name__ == "__main__":
    main(sys.argv)

