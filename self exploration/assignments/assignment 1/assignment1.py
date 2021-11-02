# Assignment 1

"""
The file `forums.pkl` is the dataset for your Assignment.

- The Assignment instructions can be found on Questrom Tools under Tests/Quizzes
- This is __individual assignment__.  You should not work or discuss this with anyone in the program.  
  
This file can be easily read into python via
"""

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import scikitplot as skplot


# read file into python environment
forums = pd.read_pickle("/Users/alexandervonschwerdtner/Desktop/BA820 - Unsupervised Machine Learning & Text Analytics/BA820-Fall-2021/assignments/assignment-01/forums.pkl")

# having a closer look at the dataset
forums.shape
forums.head()

forums.info()
forums.dtypes

# not scaling, data seems to be on the same scale (each variable has equal weight)
forums.describe().T

# add text column as the index
# forums.set_index("text", inplace=True)
# forums.head()
forums.index = forums.text
del forums['text']
forums.head()

# making sure there is no missing data
forums.isna().sum().sum()

################################
##Kmeans 
################################
KRANGE = range(2, 30)

# Declaring variables for use
inertia = []
silo = []

# Populating distortions for various clusters
for k in KRANGE:
  km = KMeans(k)
  labs = km.fit_predict(forums)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(forums, labs))

# Plotting
plt.figure(figsize=(15,5))


plt.subplot(1, 2, 1)
plt.title("Inertia")
sns.lineplot(KRANGE, inertia)

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
sns.lineplot(KRANGE, silo)

plt.show()

for i, s in enumerate(silo[:30]):
  print(i+2,s) # +2 to align num clusters with value

# # get the model
# k3 = KMeans(3)
# k3_labs = k3.fit_predict(forums)

# # metrics
# k3_silo = silhouette_score(forums, k3_labs)
# k9_ssamps = silhouette_samples(forums, k3_labs)
# np.unique(k3_labs)

# skplot.metrics.plot_silhouette(forums, k3_labs, title="KMeans - 3", figsize=(15,5))
# plt.show()

# sns.heatmap(forums, center=0, xticklabels=forums.columns)
# plt.show()

forums.k3

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

