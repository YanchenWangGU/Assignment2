import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# In this cluster part, we want to cluster daily maximun temperature, minimum
# temperature and rainfall. We ignored snowfall because it is extreme rare (only
# about 30 days out of more than 1400 days)

# get data for cluster 
# Param: df, the original dataframe
# Output: cluster data for clusting
def getClusterDataWeather(df):
    meanRain = []
    maxTemp = []
    minTemp = []
    dateArray = df.Date.unique()
    for i in range(len(dateArray)):
        meanRain.append(np.mean(df[df.Date == dateArray[i]]['PRCP']))
        maxTemp.append(np.mean(df[df.Date == dateArray[i]]['TMAX']))
        minTemp.append(np.mean(df[df.Date == dateArray[i]]['TMIN']))
    clusterData = pd.DataFrame()
    clusterData['MeanRain'] = meanRain
    clusterData['meanMaxT'] = maxTemp
    clusterData['meanMinT'] = minTemp
    return clusterData
    
# function to normalize data based on min max (linear scaling)
# Param: the data to cluster to normalize
# Output: normalized dataframe 
def normalizeData(clusterData):
    x = clusterData.values 
    # since TMAX and TMIN have int type, we need to convert them into float
    x = x.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    return normalizedDataFrame

# perform k mean cluster on a given data and k
# Param: data to be clustered and k, number of clusters 
# Output: centroids of clusters 
def getKMean(k,clusterData):
    kmeans = KMeans(n_clusters=k)
    
    normalizedDataFrame = normalizeData(clusterData)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    print('The centroids in the original data by using Kmean are:\n')
    # We want to display the centroid in the original data not the normalized 
    # one so that we can see it more clearly
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = clusterData[cluster_labels == i]
        # get the centroid in the original data by using the cluster labels and 
        # find the mean of items in the same cluster
        for j in clusterData.columns:
            centroid.append(np.mean(subData[j]))
        print(centroid,'\n')

# perform ward cluster on a given data and k
# Param: data to be clustered and k, number of clusters 
# Output: centroids of clusters     
def getWard(k,clusterData):
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    # use ward as the clustering method
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    cluster_labels = ward.fit_predict(normalizedDataFrame)
    print('The centroids in the original data by using Ward are:\n')
    # We want to display the centroid in the original data not the normalized 
    # one so that we can see it more clearly
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = sampleData[cluster_labels == i]
        for j in sampleData.columns:
            centroid.append(np.mean(subData[j]))
        print(centroid,'\n')
        
# Same as ward, we need to randomly select 10000 instances because of the memory 
# issues with the method
# Param: the data to cluster and eps
def getDBSCAN(clusterData,radius):
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    # use DBSCAN with a particular eps
    dbScan = DBSCAN(eps=radius)
    cluster_labels = dbScan.fit_predict(normalizedDataFrame)
    # Since DBSCAN is based on distance (eps), we can get number of clusters 
    # here 
    n_clusters_ = len(set(cluster_labels))
    print('Number of clusters in DBSCAN is',n_clusters_)
    print('The centroid of DBSCAN in the original data: \n')
    # Same as before, get the centroids in the original sample 
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = sampleData[cluster_labels == i]
        for j in sampleData.columns:
            centroid.append(np.mean(subData[j]))
        print(centroid,'\n')
    
# Get silhouette for k mean for a given data and k
# Param: the data to cluster and k, the number of clusters 
# Output: the mean of 5 silhouette scores 
def getKMeanScore(clusterData,k):
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    # perform k mean clustering and use the cluster label to get the silhouette score
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print('silhouette coefficient for Kmean with k =',k, 'is',silhouette_avg)
  
# Get silhouette for k mean for a given data and k
# Param: the data to cluster and k, the number of clusters 
# Output: the silhouette score   
def getWardScore(clusterData,k):
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    # perform ward
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    cluster_labels = ward.fit_predict(normalizedDataFrame)
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print('silhouette coefficient for Ward with k =',k, 'is',silhouette_avg)

# Get silhouette for DBSCAN for a given data and eps
# Param: the data to cluster and eps
# Output: the silhouette score  
def getDBSCANScore(clusterData,radius):
    sampleData = clusterData
    dbScan = DBSCAN(eps = radius)
    normalizedDataFrame = normalizeData(sampleData)
    # perform DBSCAN
    cluster_labels = dbScan.fit_predict(normalizedDataFrame)
    # Base case that if number of cluster is 1, there is no silhouette score
    # with only one cluster
    if len(set(cluster_labels)) == 1:
        print('When eps =',radius,'Number of cluster is 1 and score is not available')
        return
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print('silhouette coefficient for DBSCAN with eps =',radius, 'is',silhouette_avg)

# plot clusters with attributes PRCP, TMAX, TMIN
    
# plot k mean clustering
# Param: the data to cluster and k, the number of clusters 
def plotKmean(k,clusterData):
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    x = normalizedDataFrame.loc[:,0]
    y = normalizedDataFrame.loc[:,1]
    z = normalizedDataFrame.loc[:,2]
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    center = kmeans.cluster_centers_
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # plot points in the normalized data with clusters 
    ax.scatter(x,y,z,c=cluster_labels)
    # plot the centroids
    for i,j,k in center:
        ax.scatter(i,j,k,c='red',marker='+')
    ax.set_title('Plot of K mean with k = 2 and centroids are in red')
    ax.set_xlabel('Rain')
    ax.set_ylabel('Max Temperature')
    ax.set_zlabel('Min Temperature')

# plot ward clustering
# Param: the data to cluster and k, the number of clusters 
def plotWard(k,clusterData):
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    x = normalizedDataFrame.loc[:,0]
    y = normalizedDataFrame.loc[:,1]
    z = normalizedDataFrame.loc[:,2]
    ward = AgglomerativeClustering(n_clusters=k)
    cluster_labels = ward.fit_predict(normalizedDataFrame)
    center = []
    # find centroid in normalized data by using cluster label
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = normalizedDataFrame[cluster_labels == i]
        for j in normalizedDataFrame.columns:
            centroid.append(np.mean(subData[j]))
        center.append(np.asarray(centroid))
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # plot points in the normalized data with clusters 
    ax.scatter(x,y,z,c=cluster_labels)
    # plot the centroids 
    for i,j,k in center:
        ax.scatter(i,j,k,c='red')
    ax.set_title('Plot of Ward with number of clusters = 2 and centroids are in red')
    ax.set_xlabel('Rain')
    ax.set_ylabel('Max Temperature')
    ax.set_zlabel('Min Temperature')

# plot DBSCAN clustering
# Param: the data to cluster and eps
def plotDBSCAN(radius,clusterData):
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    x = normalizedDataFrame.loc[:,0]
    y = normalizedDataFrame.loc[:,1]
    z = normalizedDataFrame.loc[:,2]
    dbScan = DBSCAN(eps = radius)
    cluster_labels = dbScan.fit_predict(normalizedDataFrame)
    center = []
    # find centroid in normalized data by using cluster label
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = normalizedDataFrame[cluster_labels == i]
        # get the centroid in the original data. 
        for j in normalizedDataFrame.columns:
            centroid.append(np.mean(subData[j]))
        center.append(np.asarray(centroid))
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # plot points in the normalized data with clusters 
    ax.scatter(x,y,z,c=cluster_labels)
    # plot the centroids 
    for i,j,k in center:
        ax.scatter(i,j,k,c='red',marker='+')
    ax.set_title('Plot of DBSCAN with eps = .25 and centroids are in red')
    ax.set_xlabel('Rain')
    ax.set_ylabel('Max Temperature')
    ax.set_zlabel('Min Temperature')

def main(argv):
    df = pd.read_csv('weather_afterPart1.csv' , sep=',', encoding='latin1')
    clusterData = getClusterDataWeather(df)
    getKMean(2,clusterData)
    # The centroids in the original data by using Kmean are:
    # [0.14284911370514472, 81.76977950713359, 61.96952010376135] 
    # [0.09734057971014495, 50.20434782608696, 31.441304347826087] 
    
    getWard(2,clusterData)
    # The centroids in the original data by using Ward are:
    # [0.08262419871794878, 53.79086538461539, 34.58052884615385] 
    # [0.1725887652358239, 84.15182829888712, 64.70906200317965] 
    
    getDBSCAN(clusterData,.25)
    # Number of clusters in DBSCAN is 2
    # The centroid of DBSCAN in the original data: 
    # [2.716666666666667, 61.5, 50.5] 

    # [0.11779872058487537, 66.86943111720356, 47.54763536668951] 
    
    getKMeanScore(clusterData,2)
    # silhouette coefficient for Kmean with k = 2 is 0.562689103907
    
    getKMeanScore(clusterData,3)
    # silhouette coefficient for Kmean with k = 3 is 0.481463067128

    getWardScore(clusterData,2)
    # silhouette coefficient for Ward with k = 2 is 0.533078795742
    
    getWardScore(clusterData,3)
    # silhouette coefficient for Ward with k = 3 is 0.466275647485
    
    getDBSCANScore(clusterData,.15)
    # silhouette coefficient for DBSCAN with eps = 0.15 is 0.5103764991
    getDBSCANScore(clusterData,.2)
    # silhouette coefficient for DBSCAN with eps = 0.2 is 0.548403217818
    getDBSCANScore(clusterData,.25)
    # silhouette coefficient for DBSCAN with eps = 0.25 is 0.549280024936
    getDBSCANScore(clusterData,.3)
    # Number of cluster is 1, score is not available
    
    
    plotKmean(2,clusterData)
    plotWard(2,clusterData)
    plotDBSCAN(.2,clusterData)
    
if __name__ == "__main__":
    main(sys.argv)