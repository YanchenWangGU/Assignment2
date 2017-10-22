from apyori import apriori
import pandas as pd
import random

df = pd.read_csv('crashP2Original.csv' , sep=',', encoding='latin1')

def getClusterData(df):
    Total_injuries = df['MAJORINJURIES_BICYCLIST'] + df['MAJORINJURIES_DRIVER'] + \
    df['MAJORINJURIES_PEDESTRIAN']+ df['MINORINJURIES_BICYCLIST']+ df['MINORINJURIES_DRIVER']\
    + df['MINORINJURIES_PEDESTRIAN']+ df['FATAL_BICYCLIST']+ df['FATAL_DRIVER']+ df['FATAL_PEDESTRIAN']
    
    Total_involved = df['TOTAL_BICYCLES']+df['TOTAL_PEDESTRIANS'] + df['TOTAL_VEHICLES']
    clusterData = pd.DataFrame()
    clusterData['Total_injuries'] = Total_injuries
    clusterData['Total_involved'] = Total_involved
    clusterData['SPEEDING_INVOLVED'] = df['SPEEDING_INVOLVED']
    return clusterData

data = getClusterData(df)
subdata = data.sample(50000)
subdata = subdata.reset_index(drop=True)
transactions = list()
for i in range(len(subdata.index)):
    toAppend = list()
    for j in range (len(subdata.columns)):
        toAppend.append(subdata.columns[j]+str(subdata.iloc[i,j]))
     
    transactions.append(toAppend)

results = list(apriori(transactions))

for i in range(len(results)):
    print(results[i],'\n')