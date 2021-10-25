# In this session, we will explore distance calculations and their role in how we can determine 
# similarity between records.  We can use this information to segment our data into like-groups.


# resources
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html


# imports - usual suspects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for distance and h-clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

# sklearn does have some functionality too, but mostly a wrapper to scipy
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import haversine_distances 
from sklearn.preprocessing import StandardScaler

# loading dataset from GCP into environment

SQL = "SELECT * from `questrom.datasets.mtcars`"
YOUR_BILLING_PROJECT = "ba820-avs"
cars = pd.read_gbq(SQL, YOUR_BILLING_PROJECT)

# what do we have
type(cars)
cars.shape

# numpy ------ really simple dataset
x = np.array([1,3])
y = np.array([3,4])
z = np.array([2,4])
a = np.stack([x,y,z])
type(a)
a_df = pd.DataFrame(a)

# create our first distance matrix
d1 = pdist(a)
d1
type(d1)
a

# the condensed representation -> squareform
squareform(d1)

# cosine
cd = pdist(a, metric='cosine')
squareform(cd)

# cityblock
cb = pdist(a, metric='cityblock')
squareform(cb)

# sklearn
pairwise_distances(a_df, metric='euclidean')

# cars
cars.head(3)
cars.dtypes

# exercise
# model as the index
# make sure the model doesn't exist -- just a numeric dataframe
# exploration

cars.set_index('model', inplace=True)
cars.head(3)
cars.dtypes

# quick look
cars.describe().T

# keep only columns of interest
col = ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear','carb']
cars2 = cars[col]
cars2.describe().T

# cars distance --- passing in the numpy array from the pandas fram via values
cdist = pdist(cars2.values)


# squareform --- visualiye and get a sense 
sns.heatmap(squareform(cdist), cmap='Reds')
plt.show()

# our first cluster!
hc1 = linkage(cdist)
type(hc1)

dendrogram(hc1, labels=cars.index)
plt.show()

# a plot we will see later 
DIST = 80
plt.figure(figsize=(5,6))
dendrogram(hc1, 
            labels = cars.index,
            orientation = "left", 
            color_threshold = DIST)
plt.axvline(x=DIST, c='grey', lw=1, linestyle='dashed')
plt.show()

# how to identify the clusters
fcluster(hc1,2,criterion='maxclust')
cars2['cluster1'] = fcluster(hc1,2,criterion='maxclust')
cars2.head(3)

# "profiling"
cars2.cluster1.value_counts()

# how about distance assignemnt
c2 = fcluster(hc1,80,criterion='distance')
c2
cars2['cluster2'] = c2
cars2.head(3)

# different linkage methods have not been discussed
linkage(cars2.values, method=...)

# standardizing values --- put values on the same scale



# useful code snippets below ---------------------------------


# filtered cars columns
# ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear','carb']



# a plot we will see later 
# DIST = 80
# plt.figure(figsize=(5,6))
# dendrogram(hc1, 
#            labels = cars.index,
#            orientation = "left", 
#            color_threshold = DIST)
# plt.axvline(x=DIST, c='grey', lw=1, linestyle='dashed')
# plt.show()



# another advanced plot
# METHODS = ['single', 'complete', 'average', 'ward']
# plt.figure(figsize=(15,5))
# # loop and build our plot
# for i, m in enumerate(METHODS):
#   plt.subplot(1, 4, i+1)
#   plt.title(m)
#   dendrogram(linkage(cars_scaled.values, method=m),
#              labels = cars_scaled.index,
#              leaf_rotation=90,
#              leaf_font_size=10)
  
# plt.show()
