##############################################################################
## Dimension Reduction 1: Principal Components Analysis
## Learning goals:
## - application of PCA in python via sklearn
## - data considerations and assessment of fit
## - hands on data challenge = Put all of your skills from all courses together!
## - setup non-linear discussion for next session
##
##############################################################################

# installs

# notebook/colab
# ! pip install scikit-plot
# pip install scikit-plot

# imports
import numpy as np
import pandas as pd
from scipy.sparse import data
import seaborn as sns
import matplotlib.pyplot as plt

# what we need for today
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics 

import scikitplot as skplt

# color maps
from matplotlib import cm


# resources
# Seaborn color maps/palettes:  https://seaborn.pydata.org/tutorial/color_palettes.html
# Matplotlib color maps:  https://matplotlib.org/stable/tutorials/colors/colormaps.html
# Good discussion on loadings: https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html



##############################################################################
## Warmup Exercise
##############################################################################

# warmup exercise
# questrom.datasets.diamonds
# 1. write SQL to get the diamonds table from Big Query
# 2. keep only numeric columns (pandas can be your friend here!)
# 3. use kmeans to fit a 5 cluster solution
# 4. generate the silohouette plot for the solution
# 5. create a boxplot of the column carat by cluster label (one boxplot for each cluster)

PROJECT = 'ba820-avs'
SQL = "SELECT * from `questrom.datasets.diamonds`"
diamonds = pd.read_gbq(SQL, PROJECT)

diamonds.shape
diamonds.info()
diamonds.head(3)
diamonds.drop(['cut','color','clarity'], axis=1, inplace=True)

diamonds.info()

diamonds.dtypes

diamonds.describe().T

sc = StandardScaler()
diamonds_scaled = sc.fit_transform(diamonds)

# fit our first kmeans - 5 clusters
k5 = KMeans(5)
k5.fit(diamonds_scaled)

k5_labs = k5.predict(diamonds_scaled)
k5_labs

# adding cluster assignemnt to orginial dataset
diamonds['k5'] = k5_labs
diamonds.head()

k5.n_iter_

# boxplot agains the cariable carat
sns.boxplot(data=diamonds, x = 'k5', y='carat')
plt.show()

# silouhette plot
skplt.metrics.plot_silhouette(diamonds_scaled, k5.predict(diamonds_scaled), figsize=(7,7))
plt.show()


######################
## PCA
######################

PROJECT = 'ba820-avs'
SQL = "SELECT * from `questrom.datasets.judges`"
judges = pd.read_gbq(SQL, PROJECT)

judges.info()

# set index to judge
# judges.set_index('judge')
judges.index = judges.judge
del judges['judge']
judges.head(3)

# checking if we need to scale
judges.describe().T       # -- the means are verz close to each other & the st. dev are small

# looking at correlation
jc = judges.corr()
sns.heatmap(jc, cmap='Reds', center=0)
plt.show()

# fit our first model for PCA
pca = PCA()
pcs = pca.fit_transform(judges)

pcs.shape

# what is the explained variance ratio
varexp = pca.explained_variance_ratio_
type(varexp)
varexp.shape


# plot
plt.title('Explained Variance Ratio by Component')
sns.lineplot(range(1, len(varexp)+1), np.cumsum(varexp))
plt.axhline(.95)
plt.show()


# explained variiiance (not ratio)
explvar = pca.explained_variance_
type(explvar)
explvar.shape

# plot
plt.title('Explained Variance by Component')
sns.lineplot(range(1, len(explvar)+1), explvar)
plt.axhline(1)
plt.show()


##################### continuing on judges data ####################

pca.n_components_

comps = pca.components_

COLS = ['PC' + str(i) for i in range(1, len(comps)+1)]

loadings = pd.DataFrame(comps.T, columns=COLS, index=judges.columns)
loadings

# plot
sns.heatmap(loadings, cmap='vlag')
plt.show()

# matches the shape of the judges
pcs.shape
judges.shape

# put this back onto a new dataset
comps_judges = pcs[:, :2]
comps_judges.shape

j = pd.DataFrame(comps_judges, columns=['c1','c2'], index=judges.index)
j.head()

sns.scatterplot(data=j, x='c1',y='c2')
plt.show()


##############################################################################
## Code snippets for our discussion
##############################################################################

################# quick function to construct the barplot easily
# def ev_plot(ev):
#   y = list(ev)
#   x = list(range(1,len(ev)+1))
#   return x, y

# x, y = ev_plot(pca.explained_variance_)

# plt.title("Explained Variance - Eigenvalue")
# plt.bar(x=x, height=y)
# plt.axhline(y=1, ls="--")


################# loadings matrix
# component, feature
# comps = pca.components_
# COLS = ["PC" + str(i) for i in range(1, len(comps)+1)]
# loadings = pd.DataFrame(comps.T, columns=COLS, index=judges.columns)


################# categorical data for diamonds dataset plot of PC2
# dia['cut2'] = dia.cut.astype('category').cat.codes
# plt.scatter(x=dia.pc2_1, y=dia.pc2_2, c=dia.cut2, cmap=cm.Paired, alpha=.3)







##############################################################################
## PRACTICE: Data Exercise
##############################################################################

## - Diamonds data challenge in breakout rooms
## - lets start to see how we can combine UML and SML!
##
## - OBJECTIVE:  As a group, fit a regression model to the price column
## -             What is your R2? can you beat your best score?
## 
##
## 1. refit PCA to the diamonds dataset.
## 2. how many components would you select
## 3. remember!  do not include price in your columns when generating the components, we are predicint that!
## 4. Iterate!  try different models, assumptions
##
## NOTE:  we haven't covered regression in scikit learn, but its the same flow!
## Some help:  
##   
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score 

# in this case, does dimensionality reduction help us?