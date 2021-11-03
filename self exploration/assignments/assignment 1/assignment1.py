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

from scipy.spatial.distance import pdist

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import scikitplot as skplot

# color maps
from matplotlib import cm


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
##PCA
################################

# correlation matrix
fc = forums.corr()
sns.heatmap(fc, cmap="Reds", center=0)
plt.show()

# fit our first model for PCA
pca = PCA()
pcs = pca.fit_transform(forums)

pcs.shape
type(pcs)


## what is the explained variance ratio
varexp = pca.explained_variance_ratio_
type(varexp)
varexp.shape


# plot 
plt.title("Explained Variance Ratio by Component")
sns.lineplot(range(1, len(varexp)+1), varexp)
plt.show()

# cumulative view
plt.title("Explained Variance Ratio by Component")
sns.lineplot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95)
plt.show()

# explained variance (not ratio)
explvar =  pca.explained_variance_
type(explvar)
explvar.shape

plt.title("Explained Variance Ratio by Component")
sns.lineplot(range(1, len(varexp)+1), explvar)
plt.axhline(1)
plt.show()

##########

pca.n_components_

comps = pca.components_

COLS = ["PC" + str(i) for i in range(1, len(comps)+1)]

loadings = pd.DataFrame(comps.T, columns=COLS, index=forums.columns)
loadings

# plot of this
sns.heatmap(loadings, cmap="vlag")
plt.show()

# matches the shape of the forums
pcs.shape
forums.shape

# put this back onto a new dataset
comps_forums = pcs[:, :2]
comps_forums.shape

f = pd.DataFrame(comps_forums, columns = ['c1', 'c2'], index=forums.index)
f.head(3)

sns.scatterplot(data=f, x="c1", y="c2")
plt.show()


################################
##HClust
################################

# loop and build our plot
for i, m in enumerate(METHODS):
  plt.subplot(1, 4, i+1)
  plt.title(m)
  dendrogram(linkage(forums.values, method=m),
             labels = forums.index)
             #leaf_rotation=90,
             #leaf_font_size=10)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))

avg = linkage(forums.values, method="average")
dendrogram(avg,
          labels = forums.index,
          leaf_rotation=90,
          leaf_font_size=10, color_threshold=4)

plt.axhline(y=4)
plt.show()

# the clusters
hc_labs = fcluster(avg, 4, criterion="distance")

# the metrics
hc_silo = silhouette_score(forums, hc_labs)
hc_ssamps = silhouette_samples(forums, hc_labs)
np.unique(hc_labs)


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
k3 = KMeans(3)
k3_labs = k3.fit_predict(forums)

# # metrics
k3_silo = silhouette_score(forums, k3_labs)
k9_ssamps = silhouette_samples(forums, k3_labs)
np.unique(k3_labs)

#################### Comparing models via silo

# Hclust
skplot.metrics.plot_silhouette(forums, hc_labs, title="HClust", figsize=(15,5))
plt.show()

# KMEans
skplot.metrics.plot_silhouette(forums, k3_labs, title="KMeans - 3", figsize=(15,5))
plt.show()

sns.heatmap(forums, center=0, xticklabels=forums.columns)
plt.show()

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

