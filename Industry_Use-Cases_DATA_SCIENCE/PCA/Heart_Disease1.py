"""
Business Objective:
To analyze heart disease data using Hierarchical & K-Means Clustering to identify patterns and
group patients based on risk factors like age, cholesterol levels, etc. This helps in understanding 
different risk profiles for better healthcare decisions.

Constraints:
Data should be preprocessed & normalized for accurate clustering.
The optimal number of clusters needs to be determined for meaningful segmentation.
Results should be interpretable for medical insights.
Scalability: The approach should handle larger datasets efficiently.
"""
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

df=pd.read_csv("C:/DataSet/heart disease.csv")
df

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(df.iloc[:,1:])
df_norm

z=linkage(df_norm,method='complete',metric='euclidean')
#The method used is 'complete' linkage, which considers the 
#maximum distance between points in clusters. 
#The distance metric is Euclidean, which is the straight-line distance between points.
#z=This will contain the linkage matrix, which is used to generate the dendrogram.z

plt.figure(figsize=(15,8))
plt.title=('Heirarchical clustering dendogram')
plt.xlabel=('Index')
plt.ylabel=('Distance')

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering
a=AgglomerativeClustering(n_clusters=3,linkage='complete',metric='euclidean').fit(df_norm)

a.labels_
cluster_label=pd.Series(a.labels_)
df['clust']=cluster_label
df1=df.iloc[:,[5,1,2,3,4]]
df1



#normalize the data
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df_scaled=scaler.fit_transform(df)

df_scaled=pd.DataFrame(df_scaled, columns=df.columns)

sns.boxplot(df_scaled)

# Perform K-means clustering based on alcohol and malic acid
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(df_scaled[['age', 'chol']])

# Get cluster centers (centroids)
centroids = kmeans.cluster_centers_

# Create dataframes for different clusters
df_cluster1 = df_scaled[y_kmeans == 0]
df_cluster2 = df_scaled[y_kmeans == 1]
df_cluster3 = df_scaled[y_kmeans == 2]

# Plot the clusters and centroids
plt.scatter(df_cluster1['age'], df_cluster1['chol'], s=50, c='blue', label='Cluster 1')
plt.scatter(df_cluster2['age'], df_cluster2['chol'], s=50, c='red', label='Cluster 2')
plt.scatter(df_cluster3['age'], df_cluster3['chol'], s=50, c='yellow', label='Cluster 3')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='Centroids')
plt.xlabel('age')
plt.ylabel('chol')
plt.title('K-means Clustering Heart(age vs chol)')
plt.legend()
plt.show()
