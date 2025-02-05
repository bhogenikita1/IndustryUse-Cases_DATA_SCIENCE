'''
1. Perform K means clustering on the airlines dataset to obtain optimum number of clusters. 
Draw the inferences from the clusters obtained. 
Refer to EastWestAirlines.xlsx dataset.


Business Objective:
    Segment airline customers based on their balance and qualification miles.
    Identify different customer categories, such as frequent flyers, premium customers, and inactive users.
    Optimize marketing and customer retention strategies by targeting different clusters appropriately.

Constraints:
    The dataset may contain outliers affecting clustering performance.
    Feature scaling is crucial to ensure fair clustering.
    The optimal number of clusters must be determined using the Elbow Method.
    Some clusters may overlap, requiring additional feature engineering.
'''

import pandas as pd        # For reading and manipulating the dataset.
import numpy as np        # For numerical operations.
import matplotlib.pyplot as plt   #For plotting and visualizing data.
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import MinMaxScaler   #For normalizing the dataset.
from sklearn.cluster import KMeans     # For applying the K-Means clustering algorithm.

df=pd.read_excel("C:/DataSet/EastWestAirlines.xlsx")
df

# Step 1: Apply K-Means Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Balance', 'Qual_miles']])
#applies the KMeans algorithm to this data. 
#The fit part means it calculates the centroids of the 3 clusters, 

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted

# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plot the clusters
plt.scatter(df1.Balance, df1['Qual_miles'], color='orange', label='Cluster 0')
plt.scatter(df2.Balance, df2['Qual_miles'], color='blue', label='Cluster 1')
plt.scatter(df3.Balance, df3['Qual_miles'], color='green',label='Cluster 2')

# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
plt.title('K-Means Clustering')
plt.xlabel('Balance')
plt.ylabel('Qual_miles')
plt.legend()
plt.show()

#  preprocessing

scaler=MinMaxScaler()
scaler.fit(df[['Balance']])
df['Balance']=scaler.transform(df[['Balance']])

scaler.fit(df[['Qual_miles']])
df['Qual_miles']=scaler.transform(df[['Qual_miles']])

df.head()

plt.scatter(df.Balance,df['Qual_miles'])

km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(df[['Balance', 'Qual_miles']])

# Step 2: Add the cluster labels to the DataFrame
df['cluster'] = y_predicted
y_predicted
# Step 3: Visualize the clusters
# Separate the data based on cluster labels
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]


# Plot the clusters
plt.scatter(df1.Balance, df1['Qual_miles'], color='orange')#, label='Cluster 0')
plt.scatter(df2.Balance, df2['Qual_miles'], color='blue')#, label='Cluster 1')
plt.scatter(df3.Balance, df3['Qual_miles'], color='green')#,label='Cluster 2')

# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
plt.title('K-Means Clustering')
plt.xlabel('Balance')
plt.ylabel('Qual_miles')
plt.legend()
plt.show()

####### ELBOW CURVE=> NO. OF CLUSTERS .....

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage

df = pd.read_excel("C:/DataSet/EastWestAirlines.xlsx")
df

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_func(df.iloc[:,1:])
df_norm

scaler = MinMaxScaler()
df['Balance'] = scaler.fit_transform(df[['Balance']])
df['Qual_miles'] = scaler.fit_transform(df[['Qual_miles']])

# Step 1: Use the Elbow Method to find the optimal number of clusters
TWSS = []
k = list (range(2,8))
for i in k:
    km = KMeans(n_clusters=i)
    km.fit(df[['Balance', 'Qual_miles']])
    TWSS.append(km.inertia_)

# Step 2: Plot the Elbow graph
plt.figure(figsize=(10,6))
plt.plot(k, TWSS, 'ro-')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared errors (SSE)')
plt.title('Elbow Method For Optimal K')
plt.show()




















