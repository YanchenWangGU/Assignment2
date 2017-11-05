from apyori import apriori
import pandas as pd
import sys
import numpy as np

# This function will merge the car accident and weather data set by dates. 
# So that it will create a column called 'AccidentCount', which contains total 
# number of accidents for that day. It will also merge temperature, snowfall and
# rainfall by date. In particular, we took mean of three stations' oberservations
# for that day to give maximum, minimum temperature, rainfall and snowfall of
# that day. 
# Param: car accident and weather data frames 
# Output: the merged data set by date
def formNewDataframe(myData1, myData2):
    #split the data attribute to date and time
    myData1['Date'], myData1['Time'] = zip(*myData1['FROMDATE'].apply(lambda x: x.split('T')))
    myData2['Date'], myData2['Time'] = zip(*myData2['Date'].apply(lambda x: x.split('T')))
    
    # dateArray is the array that contains a list of non-repeating dates. 
    dateArray = myData2.Date.unique()
    
    # Create several empty lists for the following merge process. 
    meanRain = []
    meanSnow = []
    accidentCount = []
    
    # Use a for loop to collect data from either dataset. 
    # Then form a new dataframe that each row represent the data in a specific date. 
    # There will not be two rows representing the same date
    for i in range(len(dateArray)):
        # append the mean of the rainfall in three different location on a specific date. 
        meanRain.append(np.mean(myData2[myData2.Date == dateArray[i]]['PRCP']))
        # append the mean of the snow in three different location on a specific date.
        meanSnow.append(np.mean(myData2[myData2.Date == dateArray[i]]['SNOW']))
        # append the number of rows of accident on a specifc date. 
        accidentCount.append(myData1[myData1.Date == dateArray[i]].shape[0])

    # Create an empty dataframe
    Data = pd.DataFrame()
    
    # Add the merged attrbutes into the new dataframe. 
    Data['Date'] = dateArray                 #Date is the attribute showing date
    Data['MeanRain'] = meanRain              #MeanRain is the mean of the rain fall in three different location
    Data['AccidentCount'] = accidentCount    #AccidentCount is the number of accident in the specific day
    Data['MeanSnow'] = meanSnow              #MeanSnow is the mean of the snow fall in three different location
    # Calculate number of injuries per accident on a specific date 
    # and add a new attribute called 'injPERacci' into the dataframe
    
    # There are some obvious outliers in the Accident count. 
    Data = Data[(Data['AccidentCount'] > 10) & (Data['AccidentCount'] < 200)]
    
    # return the new merged dataframe. 
    return(Data)

# Bin accident count into binary bins called "accidentBin" and bin rain and 
# snow to a new bin called "niceWeather"
# The input variable is a dataframe. 
# This function returns the dataframe with bin attributes. 
def addBins(myData):
    # The standard of 'no rain' or 'no snow' is the data is smaller or equal 
    # to 0.1. 'rain' or 'snow' otherwise. 
    # add the bin attributes 'rainBin' and 'snowBin' into the dataframe. 
    myData.loc[myData['MeanRain'] <= 0.1, 'rainBin'] = 0
    myData.loc[myData['MeanRain'] > 0.1, 'rainBin'] = 1
    myData.loc[myData['MeanSnow'] <= 0.1, 'snowBin'] = 0
    myData.loc[myData['MeanSnow'] > 0.1, 'snowBin'] = 1
    
    # Combine the 'rainBin' and 'snowBin' to add an attribute showing the weather is nice or not
    # The standard of 'nice weather' is no rain or snow on the sepcific date
    # add the bin attribute 'niceWeather' into the dataframe. 
    myData.loc[(myData['rainBin']+ myData['snowBin']) == 0, 'niceWeather'] = 0
    myData.loc[(myData['rainBin']+ myData['snowBin']) > 0, 'niceWeather'] = 1
    
    # bin the accident count per accident into two bins (equi-depth)
    # add the bin attribute 'accidentBin' into the dataframe.
    columnlist4 = myData['AccidentCount'].tolist()
    myData.loc[myData['AccidentCount']<=np.percentile(columnlist4, 50), 'accidentBin']=0
    myData.loc[myData['AccidentCount']>np.percentile(columnlist4, 50), 'accidentBin']=1
    
    # return the resultant dataframe. 
    return(myData)


# Subset the original data to get a new data frame to run association rules
# Param: the original data 
# Output: the dataframe for this part
def getTranscationData(df):
    TranscationData = pd.DataFrame()
    # we are only using Total_injuries, Total_involved and SPEEDING_INVOLVED
    # for this part
    TranscationData['niceWeather'] = df['niceWeather']
    TranscationData['accidentBin'] = df['accidentBin']
    return TranscationData

# convert the data into transcations 
# Param: dataframe to run association rules
# Output: Transcations 
def getTranscation(TranscationData):
    transactions = list()
    # convert data into transcations 
    for i in range(len(TranscationData.index)):
        toAppend = list()
        for j in range (len(TranscationData.columns)):
            toAppend.append(TranscationData.columns[j]+str(TranscationData.iloc[i,j]))     
        transactions.append(toAppend)
    return(transactions)
    
# get indices of transcations meeting min support level 
# Param: the result from apriori and minimum support level
# Output: list of indices meeting min_supp level
def getIndicesofMinSupp(results, min_supp):
    suppSet = []
    for i in range(len(results)):
        if results[i][1] >= min_supp:
            suppSet.append(i)
    return suppSet

# get the support set meeting the min_supp level
# param: the result from apriori and indices meeting min support level
# Output: the support set with min support level 
def formSupportSet(results, supp_set):
    df = pd.DataFrame()
    df['support_set'] = ''
    df['support_level'] = ''
    for i in supp_set:
        df.loc[i] = [''.join(list(results[i][0])),results[i][1]]
    df = df.reset_index(drop=True)
    return df

# get the transcation confidence set meeting the min_supp level
# param: the result from apriori and indices meeting min support level
# Output: the transcation confidence set with min support level 
def formAssociationSet(results, supp_set):
    cols = ['From', 'To', 'conf_level']
    toAppend = []
    for i in supp_set:
        for j in range(len(results[i][2])):
            stat = list(results[i][2])[j]
            toAppend.append([''.join(list(stat[0])),''.join(list(stat[1])),stat[2]])
    df = pd.DataFrame(toAppend, columns=cols)
    return df


def main(argv):
    # Open two data sets 
    with open('crash_afterPart1.csv') as file1: 
        myData1 = pd.read_csv(file1)
    file1.closed
    
    with open('weather_afterPart1.csv') as file2: 
        myData2 = pd.read_csv(file2)
    file2.closed
    
    #adjust and merge the data to form a new dataset that is suitable to hypothesis tests
    myData = formNewDataframe(myData1, myData2)
    
    #add bins to the data
    myData = addBins(myData)
    # read data 
    df = pd.read_csv('crash_afterPart1.csv' , sep=',', encoding='latin1')
    # form data to run apriori
    df = myData
    TranscationData = getTranscationData(df)
    transactions = getTranscation(TranscationData)
    # run apriori
    results = list(apriori(transactions))
    # min_support = .05
    supp_set = getIndicesofMinSupp(results,.05)
    supportSet = formSupportSet(results,supp_set)
    associationSet = formAssociationSet(results,supp_set)
    print('The Support set and Association set for min_supp = .05\n')
    print(supportSet)
    print(associationSet)

    # min_support = .1
    supp_set = getIndicesofMinSupp(results,.1)
    supportSet = formSupportSet(results,supp_set)
    associationSet = formAssociationSet(results,supp_set)
    print('The Support set and Association set for min_supp = .1\n')
    print(supportSet)
    print(associationSet)
    
    # min_support = .25
    supp_set = getIndicesofMinSupp(results,.25)
    supportSet = formSupportSet(results,supp_set)
    associationSet = formAssociationSet(results,supp_set)
    print('The Support set and Association set for min_supp = .25\n')
    print(supportSet)
    print(associationSet)

if __name__ == "__main__":
    main(sys.argv)