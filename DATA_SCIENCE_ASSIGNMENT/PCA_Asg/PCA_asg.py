#<<<<<<<<<<<  PCA  >>>>>>>>>>>>#

'''
Problem Statement: -
Perform hierarchical and K-means clustering on the dataset. 
After that, perform PCA on the dataset and extract the first 3 principal components 
and make a new dataset with these 3 principal components as the columns. 
Now, on this new dataset, perform hierarchical and K-means clustering. 
Compare the results of clustering on the original dataset and clustering on the 
principal components dataset (use the scree plot technique to obtain the optimum 
number of clusters in K-means clustering and check if you’re getting similar results with and without PCA).
'''
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv("C:/DataSet/heart disease.csv")
df
df.isnull().sum()
sns.boxplot(df[['age','sex','cp']])

sns.boxplot(df[['trestbps','chol','fbs']])

sns.boxplot(df[['restecg','thalach','exang','oldpeak']])

sns.boxplot(df[['slope','ca','thal','target']])

outliers_columns=['trestbps','chol','fbs','thalach','oldpeak','ca','thal']


# Apply log transformation to the specified columns
for column in outliers_columns:
    # Adding a small constant to avoid issues with log(0)
    df[column] = np.log1p(df[column])

# Display the transformed data
df.head()

def remove_out(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #Remove outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df

df= remove_out(df,outliers_columns)
df
#Heirarchical clustering
'''
Normalization of Data
Before performing clustering (especially K-means), it's important to 
standardize the data. Clustering algorithms are sensitive to feature scaling, 
so normalize the dataset.
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Compute the linkage matrix
linked = linkage(df_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()


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
