from apyori import apriori
import pandas as pd
import sys

def getTranscationData(df):
    TranscationData = pd.DataFrame()
    TranscationData['Total_injuries'] = df['Total_injuries']
    TranscationData['Total_involved'] = df['Total_involved']
    TranscationData['SPEEDING_INVOLVED'] = df['SPEEDING_INVOLVED']
    return TranscationData

def getTranscation(TranscationData):
    transactions = list()
    for i in range(len(TranscationData.index)):
        toAppend = list()
        for j in range (len(TranscationData.columns)):
            toAppend.append(TranscationData.columns[j]+str(TranscationData.iloc[i,j]))     
        transactions.append(toAppend)
    return(transactions)
    
def getIndecesofMinSupp(results, min_supp):
    suppSet = []
    for i in range(len(results)):
        if results[i][1] >= min_supp:
            suppSet.append(i)
    return suppSet

def formSupportSet(results, supp_set):
    df = pd.DataFrame()
    df['support_set'] = ''
    df['support_level'] = ''
    for i in supp_set:
        df.loc[i] = [''.join(list(results[i][0])),results[i][1]]
    df = df.reset_index(drop=True)
    return df

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
    df = pd.read_csv('crash_afterPart1.csv' , sep=',', encoding='latin1')
    TranscationData = getTranscationData(df)
    transactions = getTranscation(TranscationData)
    results = list(apriori(transactions))
    supp_set = getIndecesofMinSupp(results,.05)
    supportSet = formSupportSet(results,supp_set)
    associationSet = formAssociationSet(results,supp_set)
    print('The Support set and Association set for min_supp = .05\n')
    print(supportSet)
    print(associationSet)
    
    supp_set = getIndecesofMinSupp(results,.2)
    supportSet = formSupportSet(results,supp_set)
    associationSet = formAssociationSet(results,supp_set)
    print('The Support set and Association set for min_supp = .2\n')
    print(supportSet)
    print(associationSet)
    
    supp_set = getIndecesofMinSupp(results,.5)
    supportSet = formSupportSet(results,supp_set)
    associationSet = formAssociationSet(results,supp_set)
    print('The Support set and Association set for min_supp = .5\n')
    print(supportSet)
    print(associationSet)
    
if __name__ == "__main__":
    main(sys.argv)
