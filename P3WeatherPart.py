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


def getClusterDataWeather(df):
    meanRain = []
    meanSnow = []
    maxTemp = []
    minTemp = []
    dateArray = df.Date.unique()
    for i in range(len(dateArray)):
        meanRain.append(np.mean(df[df.Date == dateArray[i]]['PRCP']))
        meanSnow.append(np.mean(df[df.Date == dateArray[i]]['SNOW']))
        maxTemp.append(np.mean(df[df.Date == dateArray[i]]['TMAX']))
        minTemp.append(np.mean(df[df.Date == dateArray[i]]['TMIN']))
    clusterData = pd.DataFrame()
    clusterData['MeanRain/Snow'] = [x + y for x, y in zip(meanSnow, meanRain)]
    clusterData['meanMaxT'] = maxTemp
    clusterData['meanMinT'] = minTemp
    return clusterData
    
    
def normalizeData(clusterData):
    x = clusterData.values 
    x = x.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    return normalizedDataFrame

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
    sampleData = clusterData
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
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    dbScan = DBSCAN(eps=radius)
    cluster_labels = dbScan.fit_predict(normalizedDataFrame)
    n_clusters_ = len(set(cluster_labels))
    print('Number of clusters in DBSCAN is',n_clusters_)
    print('The centroid of DBSCAN in the original data: \n')
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = sampleData[cluster_labels == i]
        for j in sampleData.columns:
            centroid.append(np.mean(subData[j]))
        print(centroid,'\n')
        
def getKMeanScore(clusterData,k):
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print('silhouette coefficient for Kmean with k =',k, 'is',silhouette_avg)
  
def getWardScore(clusterData,k):
    sampleData = clusterData
    normalizedDataFrame = normalizeData(sampleData)
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    cluster_labels = ward.fit_predict(normalizedDataFrame)
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print('silhouette coefficient for Ward with k =',k, 'is',silhouette_avg)

def getDBSCANScore(clusterData,radius):
    sampleData = clusterData
    dbScan = DBSCAN(eps = radius)
    normalizedDataFrame = normalizeData(sampleData)
    cluster_labels = dbScan.fit_predict(normalizedDataFrame)
    silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
    print('silhouette coefficient for DBSCAN with eps =',radius, 'is',silhouette_avg)

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
    ax.scatter(x,y,z,c=cluster_labels)
    for i,j,k in center:
        ax.scatter(i,j,k,c='red',marker='+')
    ax.set_title('Plot of K mean with k = 2 and centroids are in red')
    ax.set_xlabel('Rain or Snow')
    ax.set_ylabel('Max Temperature')
    ax.set_zlabel('Min Temperature')

def plotWard(k,clusterData):
    sampleData = clusterData
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
        ax.scatter(i,j,k,c='red')
    ax.set_title('Plot of Ward with number of clusters = 2 and centroids are in red')
    ax.set_xlabel('Rain or Snow')
    ax.set_ylabel('Max Temperature')
    ax.set_zlabel('Min Temperature')


    
def plotDBSCAN(radius,clusterData):
    sampleData = clusterData
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
    ax.set_title('Plot of DBSCAN with eps = .25 and centroids are in red')
    ax.set_xlabel('Rain or Snow')
    ax.set_ylabel('Max Temperature')
    ax.set_zlabel('Min Temperature')

def main(argv):
    df = pd.read_csv('weather_afterPart1.csv' , sep=',', encoding='latin1')
    clusterData = getClusterDataWeather(df)
    getKMean(2,clusterData)
    # The centroids in the original data by using Kmean are:
    
    # [0.14284911370514472, 81.76977950713359, 61.96952010376135] 
    
    # [0.25279951690821234, 50.20434782608696, 31.441304347826087] 
    
    getWard(2,clusterData)
    # The centroids in the original data by using Ward are:
    
    # [0.24425105485232043, 52.45189873417721, 33.81392405063291] 
     
    # [0.13652757078986577, 83.82786885245902, 63.725782414307005] 
    
    getDBSCAN(clusterData,.3)
    # Number of clusters in DBSCAN is 2
    # The centroid of DBSCAN in the original data: 

    # [7.363333333333332, 57.0, 24.0] 

    # [0.15402196293754278, 66.94337680164722, 47.62697323266987] 
    getKMeanScore(clusterData,2)
    # silhouette coefficient for Kmean with k = 2 is 0.580100911401
    getKMeanScore(clusterData,3)
    # silhouette coefficient for Kmean with k = 3 is 0.505962293498

    getWardScore(clusterData,2)
    # silhouette coefficient for Ward with k = 2 is 0.560574440509
    getWardScore(clusterData,3)
    # silhouette coefficient for Ward with k = 3 is 0.46274660151
    getDBSCANScore(clusterData,.2)
    # silhouette coefficient for DBSCAN with eps = 0.2 is 0.628078959049
    getDBSCANScore(clusterData,.25)
    # silhouette coefficient for DBSCAN with eps = 0.25 is 0.628078959049
    getDBSCANScore(clusterData,.3)
    # silhouette coefficient for DBSCAN with eps = 0.3 is 0.635907228616
    
    
    plotKmean(2,clusterData)
    plotWard(2,clusterData)
    plotDBSCAN(.3,clusterData)
    
    
    # Define outlier, we consider snow value as a outlier if it exceeded 8 inches
    # for that day. Since if snow exceeds 8 inches, school, government and businesses
    # would be closed and metro stops running