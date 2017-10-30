from apyori import apriori
import pandas as pd
import sys

# Subset the original data to get a new data frame to run association rules
# Param: the original data 
# Output: the dataframe for this part
def getTranscationData(df):
    TranscationData = pd.DataFrame()
    # we are only using Total_injuries, Total_involved and SPEEDING_INVOLVED
    # for this part
    TranscationData['Total_injuries'] = df['Total_injuries']
    TranscationData['Total_involved'] = df['Total_involved']
    TranscationData['SPEEDING_INVOLVED'] = df['SPEEDING_INVOLVED']
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
    # read data 
    df = pd.read_csv('crash_afterPart1.csv' , sep=',', encoding='latin1')
    # form data to run apriori
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
    
    # min_support = .2
    supp_set = getIndicesofMinSupp(results,.2)
    supportSet = formSupportSet(results,supp_set)
    associationSet = formAssociationSet(results,supp_set)
    print('The Support set and Association set for min_supp = .2\n')
    print(supportSet)
    print(associationSet)
    
    # min_support = .5
    supp_set = getIndicesofMinSupp(results,.5)
    supportSet = formSupportSet(results,supp_set)
    associationSet = formAssociationSet(results,supp_set)
    print('The Support set and Association set for min_supp = .5\n')
    print(supportSet)
    print(associationSet)

if __name__ == "__main__":
    main(sys.argv)
