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
from pandas.core.indexes.base import Index
import seaborn as sns
import matplotlib.pyplot as plt
from seaborn.palettes import color_palette
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



# read file into python environment
forums = pd.read_pickle("/Users/alexandervonschwerdtner/Desktop/BA820 - Unsupervised Machine Learning & Text Analytics/BA820-Fall-2021/assignments/assignment-01/forums.pkl")

# having a closer look at the dataset
forums.shape
forums.head()

forums.info()
forums.dtypes

# add text column as the index
# forums.set_index("text", inplace=True)
# forums.head()
forums.index = forums.text
del forums['text']
forums.head()

# making sure there is no missing data
forums.isna().sum().sum()

# checking for duplicates and dropping them
forums.duplicated().sum()
forums.drop_duplicates(inplace=True)
forums.duplicated().sum()

# not scaling, data seems to be on the same scale (each variable has equal weight)
forums.describe().T

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
np.sum(varexp)

# plot 
plt.title("Explained Variance Ratio by Component")
sns.lineplot(range(1, len(varexp)+1), varexp)
plt.show()

# cumulative view
plt.title("Cumulative Explained Variance Ratio by Component")
sns.lineplot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95)
plt.show()

# explained variance (not ratio) -- eigenvalue
explvar =  pca.explained_variance_
type(explvar)
explvar.shape
plt.title("Eigenvalue")
sns.lineplot(range(1, len(varexp)+1), explvar)
#plt.axhline(1)
plt.show()

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
# -------------------------------------
# referring back to the "Eigenvalue" and the "Explained Variance Ratio by Component"
# I would choose to only keep 11 components 
comps_forums = pcs[:, :11]
comps_forums.shape

f = pd.DataFrame(comps_forums, columns = ['c1', 'c2','c3','c4','c5','c6','c7','c8','c9','c10','c11'], index=forums.index)
f.head(3)

sns.scatterplot(data=f)
plt.show()


################################
## Hierarchical Clustering
################################


METHODS = ['single', 'complete', 'average', 'ward']
plt.figure(figsize=(15,5))

# loop and build our plot
for i, m in enumerate(METHODS):
  plt.subplot(1, 4, i+1)
  plt.title(m)
  dendrogram(linkage(forums, method=m),
             leaf_rotation=90,
             leaf_font_size=5)
plt.show()

# choosing ward HClust approach
plt.figure(figsize=(10, 6))

ward = linkage(forums, method="ward")
dendrogram(ward,
          leaf_rotation=90,
          leaf_font_size=10, color_threshold=4)
plt.title('HClust - Ward')
plt.axhline(y = 13, color ="black", linestyle ="--")
plt.show()

# the clusters
hc_labs = fcluster(ward, 4, criterion="maxclust")

####################################################################################

# having a closer look at the "ward" method output
ward.shape
type(ward)

# looking at the distance added at each step
len(ward)

# looking at the growth in distance added
added_dist = ward[:, 2]
added_dist

# calculate the diff at each join
penalty = np.diff(added_dist)
penalty[-4:]

# elbow method - what clustering step starts to show signs of explosion in distance
# remember, we lost one via the diff
sns.lineplot(range(1, len(penalty)+1), penalty)
plt.show()

ward_frame = pd.DataFrame(data=ward)
ward_frame.shape
ward_frame

ward_distances = ward_frame[2]
ward_distances.shape
ward_distances.mean()
ward_distances.max()
ward_distances.min()

####################################################################################


# the metrics
hc_silo = silhouette_score(forums, hc_labs)
hc_ssamps = silhouette_samples(forums, hc_labs)
np.unique(hc_labs)


################################
##Kmeans 
################################
# Declaring the range for k
KRANGE = range(2,20)

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
plt.axvline(x = 4, color ="black", linestyle ="--")
sns.lineplot(KRANGE, inertia)

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
plt.axvline(x = 4, color ="black", linestyle ="--")
sns.lineplot(KRANGE, silo)

plt.show()

for i, s in enumerate(silo[:30]):
  print(i+2,s) # +2 to align num clusters with value

# # get the model
k4 = KMeans(4)
k4_labs = k4.fit_predict(forums)

# # metrics
k4_silo = silhouette_score(forums, k4_labs)
k4_ssamps = silhouette_samples(forums, k4_labs)
np.unique(k4_labs)



#################### Comparing models via silo ###########################

# Hclust
skplot.metrics.plot_silhouette(forums, hc_labs, title="HClust - 4", figsize=(15,5))
plt.show()

# KMEans
skplot.metrics.plot_silhouette(forums, k4_labs, title="KMeans - 4", figsize=(15,5))
plt.show()

# Cluster Method selection:
# - analyizing the two cluster methods it made sense for both to go with a k=5
# - Hclust Silhouette score: 0.123 (k=4)
# - KMeans Silhouette score: 0.110 (k4)
# - Hierarchical Cluster seems to cluster the data slightly better
# - Choosing to stick with Hclust for categorizing the forum product based on theme of the discussion

# # profiling the texts
# forums['k4_labs'] = k4_labs
forums['hc_labs'] = hc_labs
forums

# # counts by cluster
# forums.k4_labs.value_counts(sort=False)
forums.hc_labs.value_counts(sort=False)

forums[forums['hc_labs']==1].index
forums[forums['hc_labs']==2].index
forums[forums['hc_labs']==3].index
forums[forums['hc_labs']==4].index

forums[forums['hc_labs']==1].describe()
forums[forums['hc_labs']==2].describe()
forums[forums['hc_labs']==3].describe()
forums[forums['hc_labs']==4].describe()

# OBSERVATIONS:
#1. cluster 1 = .....
#2. cluster 2 = .....
#3. cluster 3 = .....
#4. cluster 4 = .....