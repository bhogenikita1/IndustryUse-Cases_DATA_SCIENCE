'''
1. You have a dataset of customers with features Age, Annual Income, and 
Spending Score. You need to apply hierarchical clustering to segment these 
customers. Plot a dendrogram to decide the optimal number of clusters and 
compare it with K-Means clustering results.
Steps:
 Load the dataset.
 Apply hierarchical clustering.
 Plot a dendrogram and choose the number of clusters.
 Apply K-Means clustering with the same number of clusters.
 Compare the results.
'''


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.cluster.hierarchy import linkage

df=pd.read_csv("C:/DataSet/Mall_Customers.csv")
df.head()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
df_norm=norm_func(iloc[:,1:])
b=df_norm.describe()
b

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8))
plt.title("Hierarchical clustering dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show() 

from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',metric="euclidean").fit(df_norm)
#apply labels to the cluster
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
Univ['clust']=cluster_labels
Univ1=Univ.iloc[:,[7,1,2,3,4,5,6]]




















