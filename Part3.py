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
# Param: a dataframe 
# Output: the dataframe used for clustering 
def getClusterDataCar(df):
    clusterData = pd.DataFrame()
    clusterData['Total_injuries'] = df['Total_injuries']
    clusterData['Total_involved'] = df['Total_involved']
    clusterData['SPEEDING_INVOLVED'] = df['SPEEDING_INVOLVED']
    return clusterData
    
# function to normalize data based on min max (linear scaling)
# Param: the data to cluster to normalize
# Output: normalized dataframe 
def normalizeData(clusterData):
    x = clusterData.values 
    # since Total_involved and Total_injuries have int type, we need to convert
    # them into float
    x = x.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    return normalizedDataFrame

# We are planning to choose k to 2 becuase most values in Total_injuries are 0
# and most values in Total_involved are 2 and variations of values in these attributes
# are not very large. 
# Param: the data to cluster and k, the number of clusters 
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
    
# randomly select 10000 instances out of about 90K instances because of memory issues
# with the kernel (out of memory since it needs a lot of space to store the 
# distance between points)
# Param: the data to cluster and k, the number of clusters 
def getWard(k,clusterData):
    sampleData = clusterData.sample(10000)
    normalizedDataFrame = normalizeData(sampleData)
    # use ward as the clustering method
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    cluster_labels = ward.fit_predict(normalizedDataFrame)
    print('The centroids in the original data by using Ward are:\n')
    # Same as previous part, display the centroid in the original sample by using
    # cluster labels
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
    sampleData = clusterData.sample(10000)
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

# Because of memory issues with the kernel, we can't get the silhouette score
# for the entire data frame with about 90K instances. We decided to 
# sample it multiple times and take the mean
# Param: the data to cluster and k, the number of clusters 
# Output: the mean of 5 silhouette scores 
def getKMeanScore(clusterData,k):
    score = list()
    # sample the sample multiple times and take the mean to get the final 
    # silhouette score
    for i in range(5):
        # sample 5000 instances from the original sample and get score on the
        # 5000 samples 
        sampleData = clusterData.sample(5000)
        normalizedDataFrame = normalizeData(sampleData)
        # perform k mean clustering and use the cluster label to get the 
        # silhouette score
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(normalizedDataFrame)
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        # take mean of the five silhouette scores and report the mean
        score.append(silhouette_avg)
    print('silhouette coefficient for Kmean with k =',k, 'is',np.mean(score))
  
# Get the silhouette score of ward. Same as k mean in the previous function,
# because of the memory issue with the kernel, we decided to random choose 5000
# instances to get the silhouette score and report mean of the 5 silhouette scores
# Param: the data to cluster and k, the number of clusters 
# Output: the mean of 5 silhouette scores 
def getWardScore(clusterData,k):
    score = list()
    for i in range(5):
        # randomly select 5000 samples from the original data
        sampleData = clusterData.sample(5000)
        normalizedDataFrame = normalizeData(sampleData)
        # perform ward clustering 
        ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
        cluster_labels = ward.fit_predict(normalizedDataFrame)
        # get silhouette score from cluster labels in wards 
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        score.append(silhouette_avg)
    print('silhouette coefficient for Ward with k =',k, 'is',np.mean(score))

# Same as the previous two functions, get silhouette of DBSCAN
# Param: the data to cluster and eps in DBSCAN
# Output: the mean of 5 silhouette scores
def getDBSCANScore(clusterData,radius):
    score = list()
    for i in range(5):
        # randomly select 5000 samples from the original data
        sampleData = clusterData.sample(5000)
        # perform DBSCAN with the eps
        dbScan = DBSCAN(eps = radius)
        normalizedDataFrame = normalizeData(sampleData)
        cluster_labels = dbScan.fit_predict(normalizedDataFrame)
        # get silhouette score from cluster labels in DBSCAN 
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        score.append(silhouette_avg)
    print('silhouette coefficient for DBSCAN with eps =',radius, 'is',np.mean(score))

# Plot the clusters. with the three attributes: 'Total_injuries', 'Total_involved', 
# 'SPEEDING_INVOLVED'

# Since we couldn't plot all 90K instances in one plot, we randomly selected 
# 10000 of them and cluster on the 10000 instances and plot them
# Param: the data to cluster and k, the number of clusters
# Output: plot of clusters 
def plotKmean(k,clusterData):
    # Sample 10000 instances in the original sample 
    sampleData = clusterData.sample(10000)
    normalizedDataFrame = normalizeData(sampleData)
    # get x,y,z coordinate in the normalized data
    x = normalizedDataFrame.loc[:,0]
    y = normalizedDataFrame.loc[:,1]
    z = normalizedDataFrame.loc[:,2]
    # perform K mean cluster
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    center = kmeans.cluster_centers_
    # inintialize the plot
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # plot the original points in the plot by clusters 
    ax.scatter(x,y,z,c=cluster_labels)
    # plot the centroids and mark them in red
    for i,j,k in center:
        ax.scatter(i,j,k,c='red',marker='+')
    # set title and x,y,z labels 
    ax.set_title('Plot of K mean with k = 2 and centroids are in red')
    ax.set_xlabel('Total_injuries')
    ax.set_ylabel('Total_involved')
    ax.set_zlabel('SPEEDING_INVOLVED')

# Plot clusters with ward
# Param: the data to cluster and k, the number of clusters
# Output: plot of clusters 
def plotWard(k,clusterData):
    # Sample 10000 instances in the original sample 
    sampleData = clusterData.sample(10000)
    normalizedDataFrame = normalizeData(sampleData)
    # get x,y,z coordinate in the normalized data
    x = normalizedDataFrame.loc[:,0]
    y = normalizedDataFrame.loc[:,1]
    z = normalizedDataFrame.loc[:,2]
    # Perform ward clustering
    ward = AgglomerativeClustering(n_clusters=k)
    cluster_labels = ward.fit_predict(normalizedDataFrame)
    center = []
    # get the centroids in the normalized data
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = normalizedDataFrame[cluster_labels == i]
        for j in normalizedDataFrame.columns:
            centroid.append(np.mean(subData[j]))
        center.append(np.asarray(centroid))
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # plot the original points in the plot by clusters 
    ax.scatter(x,y,z,c=cluster_labels)
    # plot the centroids and mark them in red
    for i,j,k in center:
        ax.scatter(i,j,k,c='red',marker='+')
    # set title and x,y,z labels
    ax.set_title('Plot of Ward with number of clusters = 2 and centroids are in red')
    ax.set_xlabel('Total_injuries')
    ax.set_ylabel('Total_involved')
    ax.set_zlabel('SPEEDING_INVOLVED')

# Plot clusters with DBSCAN
# Param: the data to cluster and eps
# Output: plot of clusters  
def plotDBSCAN(radius,clusterData):
    # Sample 10000 instances in the original sample 
    sampleData = clusterData.sample(10000)
    normalizedDataFrame = normalizeData(sampleData)
    # get x,y,z coordinate in the normalized data
    x = normalizedDataFrame.loc[:,0]
    y = normalizedDataFrame.loc[:,1]
    z = normalizedDataFrame.loc[:,2]
    # Perform ward clustering
    dbScan = DBSCAN(eps = radius)
    cluster_labels = dbScan.fit_predict(normalizedDataFrame)
    # get the centroids in the normalized data. 
    center = []
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = normalizedDataFrame[cluster_labels == i]
        for j in normalizedDataFrame.columns:
            centroid.append(np.mean(subData[j]))
        center.append(np.asarray(centroid))
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    # plot the original points in the plot by clusters 
    ax.scatter(x,y,z,c=cluster_labels)
    # plot the centroids and mark them in red
    for i,j,k in center:
        ax.scatter(i,j,k,c='red',marker='+')
    # set title and x,y,z labels
    ax.set_title('Plot of DBSCAN with eps = .25 and centroids are in red')
    ax.set_xlabel('Total_injuries')
    ax.set_ylabel('Total_involved')
    ax.set_zlabel('SPEEDING_INVOLVED')

def main(argv):
    df = pd.read_csv('crash_afterPart1.csv' , sep=',', encoding='latin1')
    clusterData = getClusterDataCar(df)
    getKMean(2,clusterData)
    # The centroids in the original data by using Kmean are:
    # [0.484192037470726, 2.0995316159250588, 1.0] 
    # [0.2806376657456967, 2.0026704276152274, 0.0] 
    getWard(2,clusterData)
    # The centroids in the original data by using ward are:
    # [0.2927028128821851, 1.9990827558092132, 0.0] 
    # [0.46808510638297873, 2.0319148936170213, 1.0]
    getDBSCAN(clusterData,.25)
    # Number of clusters in DBSCAN is 3
    # The centroid of DBSCAN in the original data:
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
    # silhouette coefficient for DBSCAN with eps = 0.35 is 0.852529211487
    getDBSCANScore(clusterData,.3)
    #silhouette coefficient for DBSCAN with eps = 0.3 is 0.854888059573
    getDBSCANScore(clusterData,.25)
    # silhouette coefficient for DBSCAN with eps = 0.25 is 0.860318858337
    getDBSCANScore(clusterData,.2)
    # silhouette coefficient for DBSCAN with eps = 0.2 is 0.802258860098
    
    # From the silhouette coefficient, we can find out that when k = 2 for 
    # K mean and ward, the silhouette coefficient is higher than 3. and when
    # eps = .25, the silhouette coefficient is highest in all other 3 eps
    
    plotKmean(2,clusterData)
    
    plotWard(2,clusterData)

    plotDBSCAN(.25,clusterData)
    
    
if __name__ == "__main__":
    main(sys.argv)
