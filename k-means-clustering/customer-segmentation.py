# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
import pandas as pd

#
CUSTFILE = '/Users/fpj/Development/python/ml-with-python/k-means-clustering/data/Cust_Segmentation.csv'

# Let's set a random seed
np.random.seed(0)

# Next we will be making random clusters of points by using the make_blobs class. The make_blobs class can take in many inputs, but we will use these specific ones.
# 
# n_samples: The total number of points equally divided among clusters. <br>
# n_features: The dimension of the space to which the points are projected. <br>
# centers: The number of centers to generate, or the fixed center locations. <br>
# cluster_std: The standard deviation of the clusters. <br>
# random_state: Determines random number generation for dataset creation. <br>

# Let's start by setting the parameters for the make_blobs function
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)
#
# Now let's plot these points. We're using a scatter plot to visualize the data.
plt.scatter(X[:, 0], X[:, 1], marker='.')
plt.show()

# Set up the k0means with an init method of k-means++ and 4 clusters
k_means = KMeans(n_clusters=4, init='k-means++', n_init=12)
# Fit the k-means object to our data
k_means.fit(X)
# Let's grab labels for each point in the model using KMeans' .labels_ attribute and save it as k_means_labels
k_means_labels = k_means.labels_
print("k_means_labels: ", k_means_labels)
# We will also get the coordinates of the cluster centers using KMeans' .cluster_centers_ and save it as k_means_cluster_centers
k_means_cluster_centers = k_means.cluster_centers_
print("k_means_cluster_centers: ", k_means_cluster_centers)
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))
# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

plt.show()

# Let's repeat this for 3 clusters
k_means3 = KMeans(n_clusters=3, init='k-means++', n_init=12)
k_means3.fit(X)
k_means3_labels = k_means3.labels_
k_means3_cluster_centers = k_means3.cluster_centers_
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3_labels))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(())
ax.set_yticks(())
# show the plot
plt.show()

# Customer Segmentation with K-Means
cust_df = pd.read_csv(CUSTFILE)
print("Customer Dataframe: ", cust_df.head())
# pre-processing
df = cust_df.drop('Address', axis=1)
print("Customer Dataframe after dropping Address column: ", df.head())
# Normalize the data
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print("Normalized Dataframe: ", Clus_dataSet)
# Modeling
clusterNum = 3

# The Elbow Method
kmeans = KMeans(init='k-means++', n_clusters=clusterNum, n_init=12)
kmeans.fit(X)
labels = kmeans.labels_
print("kmeans labels: ", labels)
# Insights
df["Clus_km"] = labels
print("Customer Dataframe with cluster labels: \n", df.head(5))
# average the features in each cluster
df.groupby('Clus_km').mean()
# Let's look at the distribution of customers based on their age and income:
area = np.pi * ( X[:, 1])**2
plt.scatter(df['Age'], df['Income'], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

#
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(float))
#
plt.show()
