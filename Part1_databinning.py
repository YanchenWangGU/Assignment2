# import libraries
import pandas as pd
import numpy as np
import sys

# This function takes a dataframe, an attribute name, and a bin-name as inputs,
# It will binning the pass-in attribute with equal depth into three bins in a new attribute
# It will return the updated dataframe
def binning(df,attribute,binname):
    columnlist=df[attribute].tolist()
    # assigns the first 33% of the attribute as 1
    df.loc[df[attribute]<=np.percentile(columnlist, 33), binname]=1
    # assigns the second 33% of the attribute as 2
    df.loc[((df[attribute]>np.percentile(columnlist, 33)) & (df[attribute]<=np.percentile(columnlist, 66)), binname)]=2
    # assigns the last 33% of the attribute as 3
    df.loc[df[attribute]>np.percentile(columnlist, 66), binname]=3
    return(df)
    
def main(argv):
    # read the cleaned weather condition data
    df = pd.read_csv('weather_afterPart1.csv' , sep=',', encoding='latin1')
    
    # bin the attributes TMAX and TMIN
    binlist=['TMAX','TMIN']
    for i in binlist:
        df=binning(df, i, i+'bin')
    
    # save the dataframe with binning results 
    df.to_csv('weatherwithbin.csv',sep = ',', index = False)
    
if __name__ == "__main__":
    main(sys.argv)
    
