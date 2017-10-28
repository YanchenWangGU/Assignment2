import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
from pprint import pprint
from sklearn.metrics import silhouette_samples, silhouette_score

df = pd.read_csv('crashP2Original.csv' , sep=',', encoding='latin1')

# Define the data for cluster. We added all major, minior injuries and fatalities 
# together into a new variable called Total_injuries because most instance in major,
# minor injuries and fatalities are very small. The variance between instances will 
# be higher by adding them up and running cluster on the summation would be more 
# effective since there are bigger variance to cluster into. 
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

# Normalize data for cluster analysis 
def normalizeData(clusterData):
    x = clusterData.values 
    x = x.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalizedDataFrame = pd.DataFrame(x_scaled)
    return normalizedDataFrame

clusterData = getClusterData(df)

clusterData = clusterData[clusterData['Total_injuries'] >=4]
clusterData = clusterData[clusterData['Total_involved'] <5]
clusterData = clusterData.reset_index(drop=True)
normalizedDataFrame = normalizeData(clusterData)

def getKMean(k,normalizedDataFrame):
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(normalizedDataFrame)
    centroids = kmeans.cluster_centers_
    #pprint(cluster_labels)
    pprint(centroids)
    
getKMean(3,normalizedDataFrame)

def getWard(k,normalizedDataFrame):
    # randomly select 10000 instances out of 98K instances because of memory
    normalizedDataFrame = normalizedDataFrame.sample(10000)
    ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
    cluster_labels = ward.fit_predict(normalizedDataFrame)
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = normalizedDataFrame[cluster_labels == i]
        for j in normalizedDataFrame.columns:
            centroid.append(np.mean(subData[j]))
        print(centroid,'\n')
        
getWard(3,normalizedDataFrame)

def getDBSCAN(k,normalizedDataFrame):
    dbScan = DBSCAN(eps=.3)
    cluster_labels = dbScan.fit_predict(normalizedDataFrame)
    n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    pprint(n_clusters_)
    for i in np.unique(cluster_labels):
        centroid =list()
        subData = normalizedDataFrame[cluster_labels == i]
        for j in normalizedDataFrame.columns:
            centroid.append(np.mean(subData[j]))
        print(centroid,'\n')

def getKMeanScore(normalizedDataFrame,k):
    score = list()
    # Because of memory issues with the kernel, we can't get the silhouette_score
    # for the entire data frame with more than 90K instances. We decided to 
    # sample it multiple times and take the mean
    for i in range(5):
        normalizedDataFrame = normalizedDataFrame.sample(5000)
        kmeans = KMeans(n_clusters=4)
        cluster_labels = kmeans.fit_predict(normalizedDataFrame)
        silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
        score.append(silhouette_avg)
    print(np.mean(score))
getKMeanScore(normalizedDataFrame)
        

pca = PCA(n_components=2,whiten=True).fit(clusterData)
X_pca = pca.transform(clusterData)
kmeans = KMeans(n_clusters=k)
kmeans.fit(X_pca)
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(X_pca[:, 0], X_pca[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()


ward = AgglomerativeClustering(n_clusters=k, linkage='ward')
cluster_labels = ward.fit_predict(normalizedDataFrame)
pprint(cluster_labels)
silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
print("For n_clusters =", k, "The average silhouette_score on K mean is :", silhouette_avg)



dbScan = DBSCAN(eps=.05)
cluster_labels = dbScan.fit_predict(normalizedDataFrame)
pprint(cluster_labels)
silhouette_avg = silhouette_score(normalizedDataFrame, cluster_labels)
print("For n_clusters =", k, "The average silhouette_score on K mean is :", silhouette_avg)


clusterData = getClusterData(df)


reduced_data = PCA(n_components=2).fit_transform(clusterData)

db = DBSCAN().fit(reduced_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)


unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = reduced_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = reduced_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


def IQR(dist):
    return np.percentile(dist, 95) - np.percentile(dist, 5)

dist =clusterData['Total_involved'].tolist()
dist = clusterData['Total_injuries'].tolist()
IQR(dist)
len(clusterData[clusterData['Total_involved'] > 5])
len(clusterData[clusterData['Total_injuries'] > 4])

count13 = 0
count14 = 0
count15 = 0
count16 = 0
for i in range(len(df.index)):
    if (df['REPORTDATE'][i].split("-")[0] == '2013'):
        count13 = count13+1
    if (df['REPORTDATE'][i].split("-")[0] == '2014'):
        count14 = count14+1
    if (df['REPORTDATE'][i].split("-")[0] == '2015'):
        count15 = count15+1
    if (df['REPORTDATE'][i].split("-")[0] == '2016'):
        count16 = count16+1


