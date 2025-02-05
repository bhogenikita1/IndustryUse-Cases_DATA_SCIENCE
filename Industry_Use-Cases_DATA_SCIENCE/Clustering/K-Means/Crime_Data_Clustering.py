
'''
2.	Perform clustering for the crime data and identify the number of clusters formed and draw inferences. 
    Refer to crime_data.csv dataset.
    formed and draw inferences. Refer to crime_data.csv dataset.

Business Objective:
    Analyze crime data to classify different regions or states based on crime patterns.
    Identify high-crime, moderate-crime, and low-crime regions for better resource allocation.
    Assist law enforcement agencies in formulating crime prevention strategies.

Constraints:
    Choosing the optimal number of clusters using the Elbow Method.
    The dataset may have outliers that can influence cluster formation.
    Proper feature scaling is required to ensure fair clustering.
    K-Means assumes spherical clusters, which might not always be the 
    best representation of crime patterns
'''
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("C:/DataSet/crime_data.csv")
df

# Step 1: Apply K-Means Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Murder', 'Assault']])  
#first fits the model to the data and then immediately returns the cluster labels for each data point.

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted
# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plot the clusters
plt.scatter(df1.Murder, df1['Assault'], color='orange',label='Cluster 0')
plt.scatter(df2.Murder, df2['Assault'], color='blue', label='Cluster 1')
plt.scatter(df3.Murder, df3['Assault'], color='green',label='Cluster 2')


# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
plt.title('K-Means Clustering')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.legend()
plt.show()

#  preprocessing

scaler=MinMaxScaler()
scaler.fit(df[['Murder']])
df['Murder']=scaler.transform(df[['Murder']])

scaler.fit(df[['Assault']])
df['Assault']=scaler.transform(df[['Assault']])

df.head()

plt.scatter(df.Murder,df['Assault'])

# Step 1: Apply K-Means Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Murder', 'Assault']])

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted
# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]


# Plot the clusters
plt.scatter(df1.Murder, df1['Assault'], color='orange',label='Cluster 0')
plt.scatter(df2.Murder, df2['Assault'], color='blue', label='Cluster 1')
plt.scatter(df3.Murder, df3['Assault'], color='green',label='Cluster 2')


plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
#plt.title('K-Means Clustering')
plt.title('K-Means Clustering')
plt.xlabel('Murder')
plt.ylabel('Assault')
plt.legend()
plt.show()



####### ELBOW CURVE=> NO. OF CLUSTERS .....

scaler = MinMaxScaler()
df['Murder'] = scaler.fit_transform(df[['Murder']])
df['Assault'] = scaler.fit_transform(df[['Assault']])

# Step 1: Use the Elbow Method to find the optimal number of clusters
TWSS = []

k = list (range(2,8))
for i in k:
    km = KMeans(n_clusters=i)
    km.fit(df[['Murder', 'Assault']])
    TWSS.append(km.inertia_)

# Step 2: Plot the Elbow graph
plt.figure(figsize=(10,6))
plt.plot(k, TWSS, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared errors (SSE)')
plt.title('Elbow Method For Optimal K')
plt.show()


