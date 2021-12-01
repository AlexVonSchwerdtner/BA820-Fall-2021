# Learning goals:
## Expand on Distance and now apply Kmeans
## - Kmeans applications
## - Evaluate cluster solutions 
## - hands on with Kmeans and other approaches

# resources
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf
# https://scikit-learn.org/stable/modules/clustering.html#dbscan

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

# what we need for today
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import scikitplot as skplt



# Review of distance h Clustering

# dataset urls:
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MedGPA.csv


election = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv")
election.shape
election.head(3)
election.describe

# add state abbreviation as the index
election.set_index("Abr", inplace=True)
election.head()

# keep just the numerical columns
election.drop('State', axis=1, inplace=True)

election2 = election.loc[:,'Income':'Dem.Rep',]
election2.head()

# scaler
scaler = StandardScaler()
election_scaled = scaler.fit_transform(election2)
type(election_scaled)

# cluster
hc1 = linkage(election_scaled, method='complete')


# create the plot
plt.figure(figsize=(15,5))
dendrogram(hc1)
plt.show()

# create 4 clusters
cluster = fcluster(hc1, 4,criterion='maxclust')
cluster

# add cluster column to the data
election['cluster'] = cluster

# simple profile of a cluster
election.groupby('cluster')['ObamaWin'].mean()
election

# counting the amount of records in each cluster
election.cluster.value_counts()






###### K-Means Clustering ######

PROJECT = 'ba820-avs'   
SQL = "SELECT * from `questrom.datasets.judges`"
judges = pd.read_gbq(SQL, PROJECT)

judges.shape

judges.sample(3)

judges.info()

# column lowercase
judges.columns = judges.columns.str.lower()

# the judge as the index
judges.set_index('judge',inplace=True)

# the datatypes
judges.dtypes

# let's describe the dataset
judges.describe().T

# fit our first kmeans - 3 clusters
k3 = KMeans(3)
k3.fit(judges)

k3_labs = k3.predict(judges)
k3_labs

# how many interations were needed for convergence
k3.n_iter_

# put these back onto the original dataset
judges['k3'] = k3_labs
judges.sample(3)

# kmeans with 5 clusters
j = judges.copy()
del j['k3']

k5 = KMeans(5,  n_init=100)
judges['k5'] = k5.fit_predict(j)
judges.sample(3)

# start to profile/leran about our cluster
judges.k3.value_counts()
judges.k5.value_counts()

# groupby -- looking at each column by cluster
#del judges['k5']
#judges.groupby('k3').mean()
del judges['k3']
judges.groupby('k5').mean()

# extract cluster centers 
## clusters, #features
k5.cluster_centers_

test_centres = k5.cluster_centers_
test_centres.shape

judges.k5.value_counts()
k5_profile = judges.groupby('k5').mean()

sns.heatmap(k5_profile)
plt.show()

### goodness of fit ### (minimize this value)
k3.inertia_
k5.inertia_

## exercise
# how would you iterate over solution form 2- 10 on the judges dataset
# fit the cluster solution for 2 - 10
# and how would you evaluate/inspect the inertia for solutions


KRANGE = range(2, 11)
# containers
ss = []
for k in KRANGE:
  km = KMeans(k)
  lab = km.fit_predict(j)
  ss.append(km.inertia_)
ss

sns.lineplot(KRANGE, ss)
plt.show()

# silo score from sklearn -- metrics module

silo_overall = metrics.silhouette_score(j, k5.predict(j))
silo_overall

# silo samples

silo_samples = metrics.silhouette_samples(j , k5.predict(j))
type(silo_samples)
silo_samples.shape

# plotting this up

skplt.metrics.plot_silhouette(j, k5.predict(j), figsize=(7,7))
plt.show()


# useful code snippets below ---------------------------------

# scale the data
# el_scaler = StandardScaler()
# el_scaler.fit(election)
# election_scaled = el_scaler.transform(election)

# kmeans
# k5 = KMeans(5,  n_init=100)
# judges['k5'] = k5.fit_predict(j)


# k5_centers = k5.cluster_centers_
# sns.scatterplot(data=judges, x="CONT", y="INTG", cmap="virdis", hue="k5")
# plt.scatter(k5_centers[:,0], k5_centers[:,1], c="g", s=100)


# KRANGE = range(2, 30)
# # containers
# ss = []
# for k in KRANGE:
#   km = KMeans(k)
#   lab = km.fit_predict(j)
#   ss.append(km.inertia_)


# skplt.metrics.plot_silhouette(j, k5.predict(j), figsize=(7,7))

