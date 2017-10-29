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


# Define the data for cluster. We added all major, minior injuries and fatalities 
# together into a new variable called Total_injuries because most instance in major,
# minor injuries and fatalities are very small. The variance between instances will 
# be higher by adding them up and running cluster on the summation would be more 
# effective since there are bigger variance to cluster into. 


def getClusterData(df):
    clusterData = pd.DataFrame()
    clusterData['Total_injuries'] = df['Total_injuries']
    clusterData['Total_involved'] = df['Total_involved']
    clusterData['SPEEDING_INVOLVED'] = df['SPEEDING_INVOLVED']
    return clusterData

# Normalize data for cluster analysis using min max scaler
def normalizeData(clusterData):
    x = clusterData.values 
    x = x.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    return normalizedDataFrame

# We are planning to choose k to 2 becuase most values in Total_injuries are 0
# and most values in Total_involved are 2 and variations of values in these attributes
# are not large at all. 
def getKMean(k,clusterData):
    kmeans = KMeans(n_clusters=k)
    normalizedDataFrame = normalizeData(clusterData)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    print('The centroids in the original data by using Kmean are:\n')
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = clusterData[cluster_labels == i]
        # get the centroid in the original data. 
        for j in clusterData.columns:
            centroid.append(np.mean(subData[j]))
        print(centroid,'\n')
    
def getWard(k,clusterData):
    # randomly select 10000 instances out of 98K instances because of memory
    sampleData = clusterData.sample(10000)
    normalizedDataFrame = normalizeData(sampleData)
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    cluster_labels = ward.fit_predict(normalizedDataFrame)
    print('The centroids in the original data by using Ward are:\n')
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = sampleData[cluster_labels == i]
        for j in sampleData.columns:
            centroid.append(np.mean(subData[j]))
        print(centroid,'\n')
        
def getDBSCAN(clusterData,radius):
    sampleData = clusterData.sample(10000)
    normalizedDataFrame = normalizeData(sampleData)
    dbScan = DBSCAN(eps=radius)
    cluster_labels = dbScan.fit_predict(normalizedDataFrame)
    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print('Number of clusters in DBSCAN is',n_clusters_)
    print('The centroid of DBSCAN in the original data: \n')
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = sampleData[cluster_labels == i]
        for j in sampleData.columns:
            centroid.append(np.mean(subData[j]))
        print(centroid,'\n')

def getKMeanScore(clusterData,k):
    score = list()
    # Because of memory issues with the kernel, we can't get the silhouette_score
    # for the entire data frame with more than 90K instances. We decided to 
    # sample it multiple times and take the mean
    for i in range(5):
        sampleData = clusterData.sample(5000)
        normalizedDataFrame = normalizeData(sampleData)
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(normalizedDataFrame)
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        score.append(silhouette_avg)
    print('silhouette coefficient for Kmean with k =',k, 'is',np.mean(score))
  
def getWardScore(clusterData,k):
    score = list()
    # Because of memory issues with the kernel, we can't get the silhouette_score
    # for the entire data frame with more than 90K instances. We decided to 
    # sample it multiple times and take the mean
    for i in range(5):
        sampleData = clusterData.sample(5000)
        normalizedDataFrame = normalizeData(sampleData)
        ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = ward.fit_predict(normalizedDataFrame)
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        score.append(silhouette_avg)
    print('silhouette coefficient for Ward with k =',k, 'is',np.mean(score))

def getDBSCANScore(clusterData,radius):
    score = list()
    # Because of memory issues with the kernel, we can't get the silhouette_score
    # for the entire data frame with more than 90K instances. We decided to 
    # sample it multiple times and take the mean
    for i in range(5):
        dbScan = DBSCAN(eps = radius)
        sampleData = clusterData.sample(5000)
        normalizedDataFrame = normalizeData(sampleData)
        cluster_labels = dbScan.fit_predict(normalizedDataFrame)
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        score.append(silhouette_avg)
    print('silhouette coefficient for DBSCAN with eps =',radius, 'is',np.mean(score))

def plotKmean(k,clusterData):
    sampleData = clusterData.sample(10000)
    normalizedDataFrame = normalizeData(sampleData)
    x = normalizedDataFrame.loc[:,0]
    y = normalizedDataFrame.loc[:,1]
    z = normalizedDataFrame.loc[:,2]
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    center = kmeans.cluster_centers_
    fignum = 1
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(x,y,z,c=cluster_labels)
    for i,j,k in center:
        ax.scatter(i,j,k,c='red',marker='+')
    ax.set_title('Plot of K mean with k = 2 and centroids are in red')
    ax.set_xlabel('Total_injuries')
    ax.set_ylabel('Total_involved')
    ax.set_zlabel('SPEEDING_INVOLVED')

def plotWard(k,clusterData):
    sampleData = clusterData.sample(10000)
    normalizedDataFrame = normalizeData(sampleData)
    x = normalizedDataFrame.loc[:,0]
    y = normalizedDataFrame.loc[:,1]
    z = normalizedDataFrame.loc[:,2]
    ward = AgglomerativeClustering(n_clusters=k)
    cluster_labels = ward.fit_predict(normalizedDataFrame)
    center = []
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = normalizedDataFrame[cluster_labels == i]
        # get the centroid in the original data. 
        for j in normalizedDataFrame.columns:
            centroid.append(np.mean(subData[j]))
        center.append(np.asarray(centroid))
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(x,y,z,c=cluster_labels)
    for i,j,k in center:
        ax.scatter(i,j,k,c='red',marker='+')
    ax.set_title('Plot of Ward with number of clusters = 2 and centroids are in red')
    ax.set_xlabel('Total_injuries')
    ax.set_ylabel('Total_involved')
    ax.set_zlabel('SPEEDING_INVOLVED')

    
def plotDBSCAN(radius,clusterData):
    sampleData = clusterData.sample(10000)
    normalizedDataFrame = normalizeData(sampleData)
    x = normalizedDataFrame.loc[:,0]
    y = normalizedDataFrame.loc[:,1]
    z = normalizedDataFrame.loc[:,2]
    dbScan = DBSCAN(eps = radius)
    cluster_labels = dbScan.fit_predict(normalizedDataFrame)
    center = []
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = normalizedDataFrame[cluster_labels == i]
        # get the centroid in the original data. 
        for j in normalizedDataFrame.columns:
            centroid.append(np.mean(subData[j]))
        center.append(np.asarray(centroid))
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(x,y,z,c=cluster_labels)
    for i,j,k in center:
        ax.scatter(i,j,k,c='red',marker='+')
    ax.set_title('Plot of DBSCAN with number of clusters = 2 and centroids are in red')
    ax.set_xlabel('Total_injuries')
    ax.set_ylabel('Total_involved')
    ax.set_zlabel('SPEEDING_INVOLVED')


def main(argv):
    df = pd.read_csv('crash_afterPart1.csv' , sep=',', encoding='latin1')
    clusterData = getClusterData(df)
    getKMean(2,clusterData)
    # The centroids in the original data are:
    # [0.484192037470726, 2.0995316159250588, 1.0] 
    # [0.2806376657456967, 2.0026704276152274, 0.0] 
    getWard(2,clusterData)
    # The centroids in the original data are:
    # [0.2927028128821851, 1.9990827558092132, 0.0] 
    # [0.46808510638297873, 2.0319148936170213, 1.0]
    getDBSCAN(clusterData,.25)
    # We use EPS = 0.25 because when EPS = .25, the silhouette_score is highest
    # [6.0, 4.0, 0.0] 
    # [0.29011841567986935, 1.9972437729685586, 0.0]     
    # [0.4187192118226601, 2.2167487684729066, 1.0]  
    # Compare with different k value 
    getKMeanScore(clusterData,2)
    # silhouette coefficient for Kmean with k= 2 is 0.82874215613
    getKMeanScore(clusterData,3)
    # silhouette coefficient for Kmean with k= 2 0.772950271498
    getWardScore(clusterData,2)    
    # silhouette coefficient for Ward with k = 2 is 0.82497653045
    getWardScore(clusterData,3)
    # silhouette coefficient for Ward with k = 3 is 0.780263673914
    getDBSCANScore(clusterData,.35)
    # silhouette coefficient for DBSCAN with eps = 0.35 is 0.84040247202
    getDBSCANScore(clusterData,.25)
    # silhouette coefficient for DBSCAN with eps = 0.25 is 0.850147523982
    
    plotKmean(2,clusterData)
    
    plotWard(2,clusterData)  

    plotDBSCAN(.25,clusterData)  
    
if __name__ == "__main__":
    main(sys.argv)
