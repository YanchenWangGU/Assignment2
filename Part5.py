#import the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interp
from scipy.stats import ttest_ind
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import sys

#Merge two datasets into one dataframe 
#The input variable is two datasets without noise from the previous part. 
#This function returns the merged dataframe. 
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
    maxTemp = []
    minTemp = []
    totalInjury = []
    
    # Use a for loop to collect data from both dataset. 
    # Then form a new dataframe that each row represent the data in a specific date. 
    # There will not be two rows representing the same date
    for i in range(len(dateArray)):
        
        # append the mean of the rainfall in three different location on a specific date. 
        meanRain.append(np.mean(myData2[myData2.Date == dateArray[i]]['PRCP']))
        # append the mean of the snow in three different location on a specific date.
        meanSnow.append(np.mean(myData2[myData2.Date == dateArray[i]]['SNOW']))
        # append the number of rows of accident on a specifc date this value 
        # represents total number of accidents for that day. 
        accidentCount.append(myData1[myData1.Date == dateArray[i]].shape[0])
        # append the mean of the maximum temperature in three different location 
        # on a specific date.
        maxTemp.append(np.mean(myData2[myData2.Date == dateArray[i]]['TMAX']))
        # append the mean of the minimum temperature in three different location on a specific date.
        minTemp.append(np.mean(myData2[myData2.Date == dateArray[i]]['TMIN']))
        # append the sum of injuries in each car accident on a specific date. 
        totalInjury.append(np.sum(myData1[myData1.Date == dateArray[i]]['Total_injuries']))
        
    # Create an empty dataframe
    Data = pd.DataFrame()
    
    # Add the merged attrbutes into the new dataframe. 
    Data['Date'] = dateArray                 #Date is the attribute showing date
    Data['MeanRain'] = meanRain              #MeanRain is the mean of the rain fall in three different location
    Data['AccidentCount'] = accidentCount    #AccidentCount is the number of accident in the specific day
    Data['MeanSnow'] = meanSnow              #MeanSnow is the mean of the snow fall in three different location
    Data['meanMaxT'] = maxTemp               #meanMaxT is the mean of the maximum temperature in three different location
    Data['meanMinT'] = minTemp               #meanMinT is the mean of the minimum temperature in three different location
    Data['totalInjury'] = totalInjury        #totalInjusry is the total number of injuries in a specific day
    # Calculate number of injuries per accident on a specific date 
    # and add a new attribute called 'injPERacci' into the dataframe
    Data['injPERacci'] = Data['totalInjury']/Data['AccidentCount']
    
    # take a look at the data and check for outliers
    #print(Data.sort_values('AccidentCount'))
    
    # There are some obvious outliers in the Accident count and we want to remove them. 
    Data = Data[(Data['AccidentCount'] > 10) & (Data['AccidentCount'] < 200)]
    
    #return the new merged dataframe. 
    return(Data)

# Divide some of the attributes into bins and add new bin attributes into the dataframe. 
# The input variable is a dataframe. 
# This function returns the dataframe with bin attributes. 
def addBins(myData):
    
    # Turn the 'meanMaxT' column into a list in order to get the percentiles. 
    # Bin the mean maximum temperature by 0 - 33, 33 - 66 and 66 - 100 percentiles
    # (equi-depth bining). 
    # 'meanMaxT' is divided into three bins labeled 1,2,3 respectively. 
    # add the bin attribute 'maxTbin' into the dataframe. 
    columnlist1=myData['meanMaxT'].tolist()
    myData.loc[myData['meanMaxT']<=np.percentile(columnlist1, 33), 'maxTbin']=1
    myData.loc[((myData['meanMaxT']>np.percentile(columnlist1, 33)) & (myData['meanMaxT']<=np.percentile(columnlist1, 66)), 'maxTbin')]=2
    myData.loc[myData['meanMaxT']>np.percentile(columnlist1, 66), 'maxTbin']=3
    
    # Bin the mean minimum temperature by 33 and 66 percentiles. 
    # 'meanMinT' is divided into three bins labeled 1,2,3 respectively. 
    # add the bin attribute 'minTbin' into the dataframe. 
    columnlist1=myData['meanMinT'].tolist()
    myData.loc[myData['meanMinT']<=np.percentile(columnlist1, 33), 'minTbin']=1
    myData.loc[((myData['meanMinT']>np.percentile(columnlist1, 33)) & (myData['meanMinT']<=np.percentile(columnlist1, 66)), 'minTbin')]=2
    myData.loc[myData['meanMinT']>np.percentile(columnlist1, 66), 'minTbin']=3
    
    # bin the snow and rainfall into binary bins (snow or not, rain or not). 
    # The standard of 'no rain' or 'no snow' is the data is smaller or equal to 
    # 0.1 and 'rain' or 'snow' otherwise. 
    # add the bin attributes 'rainBin' and 'snowBin' into the dataframe. 
    myData.loc[myData['MeanRain'] <= 0.1, 'rainBin'] = 0
    myData.loc[myData['MeanRain'] > 0.1, 'rainBin'] = 1
    myData.loc[myData['MeanSnow'] <= 0.1, 'snowBin'] = 0
    myData.loc[myData['MeanSnow'] > 0.1, 'snowBin'] = 1
    
    #Combine the 'rainBin' and 'snowBin' to add an attribute showing the weather is nice or not
    #The standard of 'nice weather' is no rain or snow on the sepcific date
    #add the bin attribute 'niceWeather' into the dataframe. 
    myData.loc[(myData['rainBin']+ myData['snowBin']) == 0, 'niceWeather'] = 0
    myData.loc[(myData['rainBin']+ myData['snowBin']) > 0, 'niceWeather'] = 1
    
    #bin the injury into two bins (below median and above median)
    #add the bin attribute 'injuryBin' into the dataframe. 
    columnlist2 = myData['totalInjury'].tolist()
    myData.loc[myData['totalInjury']<=np.percentile(columnlist2, 50), 'injuryBin']=1
    myData.loc[myData['totalInjury']>np.percentile(columnlist2, 50), 'injuryBin']=-1
    
    #bin the injury per accident into two bins (below median and above median)
    #add the bin attribute 'rateBin' into the dataframe. 
    columnlist3 = myData['injPERacci'].tolist()
    myData.loc[myData['injPERacci']<=np.percentile(columnlist3, 50), 'rateBin']=1
    myData.loc[myData['injPERacci']>np.percentile(columnlist3, 50), 'rateBin']=-1
    
    #bin the accident count per accident into two bins (below median and above median)
    #add the bin attribute 'accidentBin' into the dataframe.
    columnlist4 = myData['AccidentCount'].tolist()
    myData.loc[myData['AccidentCount']<=np.percentile(columnlist4, 50), 'accidentBin']=-1
    myData.loc[myData['AccidentCount']>np.percentile(columnlist4, 50), 'accidentBin']=1
    
    #return the resultant dataframe. 
    return(myData)

#This is a t test for the first hypothesis
#The input variable is the dataframe needed for the test. 
#This function prints out the result of the t test. 
#This function has no returns. 
def h1Test(h1Data):
    
    #Divide the 'AccidentCount' into two groups for nice weather and bad weather respectively. 
    #The grouping is done by the attribute 'niceWeather'. 
    niceWeatherAccident = h1Data[h1Data['niceWeather'] == 0]['AccidentCount']
    badWeatherAccident = h1Data[h1Data['niceWeather'] == 1]['AccidentCount']
    
    #get the t and p value using ttest_ind
    t, p = ttest_ind(badWeatherAccident,niceWeatherAccident)
    
    #print out the p value and the mean of both groups in otder to get an overall image of the result. 
    print('T Test:')
    print('P value: ',p)
    print('mean for nice weather: ', np.mean(niceWeatherAccident))
    print('mean for bad weather: ', np.mean(badWeatherAccident))

#This is a linear regression test for the second hypothesis. 
#The input variable is the dataframe needed for the test. 
#This function prints out the result for the linear regression. 
#This function has no returns. 
def h2Test(myData):
    
    # X is the attribute 'meanMaxT'. Convert it into an array. 
    x = np.array(myData['meanMaxT'])
    # Y is the attribute 'AccidentCount'. Convet it into an array. 
    y = np.array(myData['AccidentCount'])
    
    #fit x and y into linear regression. 
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    
    #Print out the slope, r-squared and p value in otder to get an overall image of the result. 
    print('Linear Regression: ')
    print('Slope: ', slope)
    print("r-squared:", r_value**2)
    print('p value: ', p_value)

#The h3 Test contains SVM and K Nearest Neighbor on selected unbinned data. 
#For the input variables, myData is the dataframe needed, label is the list of data attributes, toPredict is the target attribute. 
#This function prints out the accracy score, confusion matrix and ROC curve. 
#This function has no returns. 
def h3Test(myData,label, toPredict):
    
    #append the target attribut at the end of label list
    #and create a new dataframe with only the attributes in the list
    #the target attribute would stay at the last column of the new dataframe. 
    label.append(toPredict[0])
    newData = myData[label]
    valueArray = newData.values
    
    #divide the dataframe into X and Y data arrays. 
    # X is the non-target, unbinned data
    X = valueArray[:,0:len(label)-1]
    #X is normalized
    X = preprocessing.normalize(X)
    #Y is the binned target data
    Y = valueArray[:,len(label)-1]
    
    #set the test size and seed
    test_size = 0.20
    seed = 7
    
    #split X and Y into train set and validate set respectively. 
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    #set the number of folds and seed
    num_folds = 10
    seed = 100
    #the score in this test is accuracy score. 
    scoring = 'accuracy'
    
    #append both SVM and KNN into the models list. 
    models = []
    models.append(('SVM', SVC(probability=True)))
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 50)))
    
    results = []
    names = []
    #Use a for loop to do the cross validation, test, print out confusion matrix and draw ROC curve. 
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        #cross validation
        cv_results = cross_val_score(model, X_train , Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        #print the cross validation result
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        #call the fitANDprint function to print the confusion matrix and report for the classification result. 
        fitANDprint(name, model, X_train, Y_train, X_validate, Y_validate)
        #call the drawROC function to draw the ROC curve. 
        drawROC(name, model, X_train, Y_train, X_validate, Y_validate)
 
#The h4 Test contains Decision Tree, Random Forest and Naive Bayes on selected binned data. 
#For the input variables, myData is the dataframe needed, label is the list of data attributes, toPredict is the target attribute. 
#This function prints out the accracy score, confusion matrix and ROC curve. 
#This function has no returns.
def h4Test(myData,label, toPredict):
    
    #append the target attribut at the end of label list
    #and create a new dataframe with only the attributes in the list
    #the target attribute would stay at the last column of the new dataframe.
    label.append(toPredict[0])
    newData = myData[label]
    valueArray = newData.values
    
    #divide the dataframe into X and Y data arrays. 
    # X is the non-target binned data
    # Since the data is already binned, X does not need to be normalized. 
    X = valueArray[:,0:len(label)-1]
    # Y is the binned target attribute
    Y = valueArray[:,len(label)-1]
    
    #set the test size and seed
    test_size = 0.20
    seed = 7
    
    #split X and Y into train set and validate set respectively. 
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=test_size, random_state=seed)
    
    #set the number of folds and seed
    num_folds = 10
    seed = 100
    #the score in this test is accuracy score.
    scoring = 'accuracy'
      
    #append Decision Tree, Random Forest and Naive Bayes into the models list.
    models = []
    models.append(('Decision Tree', DecisionTreeClassifier()))
    models.append(('Random Forest', RandomForestClassifier()))
    models.append(('Naive Bayes', GaussianNB()))
    
    results = []
    names = []
    #Use a for loop to do the cross validation, test, print out confusion matrix and draw ROC curve. 
    for name, model in models:
        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=False)
        #cross validation
        cv_results = cross_val_score(model, X_train , Y_train, cv=kfold, scoring=scoring)
        #print the cross validation result
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        #call the fitANDprint function to print the confusion matrix and report for the classification result.
        fitANDprint(name, model, X_train, Y_train, X_validate, Y_validate)
        #call the drawROC function to draw the ROC curve.
        drawROC(name, model, X_train, Y_train, X_validate, Y_validate)
       
#This function fit the train sets with the given method and test the test set.  
#The input variables are name of the classifier, the classifier call, train and test sets. 
#This function print out the confusion matrix, accuracy score for the classification and the overall report of the performance of the test. 
#This function has no returns  
def fitANDprint(name, mode, X_train, Y_train, X_validate, Y_validate):
    
    #model is the input classifier. 
    model = mode
    #fit the train models
    model.fit(X_train, Y_train)
    #make a prediction. 
    predictions = model.predict(X_validate)
    
    #print out the results
    print(name, 'Accuracy Score')
    print(accuracy_score(Y_validate, predictions))
    print('Confusion Matrix')
    print(confusion_matrix(Y_validate, predictions))
    print(classification_report(Y_validate, predictions))

#This function draws the ROC curve for a specific test. 
#The input variables are name of the classifier, the classifier call, train and test sets.
#This function has no returns
def drawROC(name, mode, X_train, Y_train, X_validate, Y_validate): 
    
    #model is the inout classifier
    model = mode
    #create two empty lists for tpr and auc
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    #get the list of false positive rate and true positive rate. 
    probas_ = model.fit(X_train, Y_train).predict_proba(X_validate)
    fpr, tpr, thresholds = roc_curve(Y_validate, probas_[:, 1])
    #interpret the rates to get true positive rates
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    #get the auc value
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    #start ploting
    #first plot the Random Guess line. 
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    
    #Then plot the ROC curve with false positive rate and true positive rate. 
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'ROC Curve', lw=2, alpha=.8)
    
    #adjust the design of the plot
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    
    #show the plot on screen. 
    plt.show()
    
#main function
#This function contains the whole general process for the hypothesis part.    
def main(argv): 
    
    #import the data as dataframe
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
    
    #The first hypothesis is that the nice weather will increase the number of car accidents. 
    #H0: the mean of number of accidents are the same on nice and bad weather days. 
    #HA: the mean of number of accidents are not the same on nice and bad weather days. 
    h1Test(myData)
    #Therefore H0 is rejected and our hypothesis is supported by our data. 
    
    #The second hypothesis is that there is a linear relationship between maximum temperature and the number of car accidents. 
    h2Test(myData)
    #Therefore H0 cannot be rejected and there is not a linear relationship bewtween maximum temperature and the number of car accidents. 
    
    #The third hypothesis is 
    #Among maximum temperature, minimum temperature, rainfall, snow and accident count, 
    #which combination of these factors affect number of injuries the most? 
    h3Test(myData,['MeanRain','meanMaxT','AccidentCount'], ['rateBin'])
    #The conclusion is that rainfall, maximum temperature and count of accident affect the rate the most. 
    
    #h3Test(myData,['meanMaxT','MeanRain','MeanSnow'], ['rateBin'])
    #h3Test(myData,['meanMinT','MeanRain','MeanSnow'], ['rateBin'])
    #h3Test(myData,['MeanRain','MeanSnow','AccidentCount'], ['rateBin'])
    #h3Test(myData,['MeanRain','MeanSnow'], ['rateBin'])
    #h3Test(myData,['MeanRain','MeanSnow', 'meanMaxT','AccidentCount'], ['rateBin'])
    
    #The foutrh hypothesis is among all the factors, maximum temperature affect the total injury the most. 
    h4Test(myData,['maxTbin','rainBin','snowBin'], ['injuryBin'])
    #The conclusion is that maximum temperature, rainfall and snow affect the total injury the most.   
    
    #h4Test(myData,['minTbin','maxTbin','niceWeather'], ['injuryBin'])
    #h4Test(myData,['maxTbin','snowBin'], ['injuryBin'])
    #h4Test(myData,['rainBin','snowBin'], ['injuryBin'])

if __name__ == "__main__":
    main(sys.argv)