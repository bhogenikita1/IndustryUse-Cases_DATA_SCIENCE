
'''
5. You are given a dataset with two numerical features Height and Weight. 
Your goal is to cluster these people into 3 groups using K-Means clustering. 
After clustering, you will visualize the clusters and their centroids.
 Load the dataset (or generate random data for practice).
 Apply K-Means clustering with k = 3.
 Visualize the clusters and centroids.
 Experiment with different values of k and see how the clustering chang

Business Objective:
Cluster individuals based on their height and weight to identify patterns in body types.
Useful for applications in healthcare, fitness, and retail for personalized recommendations.

Constraints:
Optimal number of clusters (k) needs to be carefully chosen (Elbow Method recommended).
K-Means assumes spherical and equally sized clusters, which may not always hold true.
Proper feature scaling is required to ensure unbiased clustering.
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv("C:/DataSet/HeightWeight.csv")

# Display first 5 rows
print(df.head())

# Feature Scaling (Normalize Height & Weight)
scaler = MinMaxScaler()
df[['Height(Inches)', 'Weight(Pounds)']] = scaler.fit_transform(df[['Height(Inches)', 'Weight(Pounds)']])

# Applying K-Means clustering with k=3
k = 3
km = KMeans(n_clusters=k, n_init=10, random_state=42)
df['Cluster'] = km.fit_predict(df[['Height(Inches)', 'Weight(Pounds)']])

# Get Cluster Centers (Centroids)
centroids = km.cluster_centers_

# Plot Clusters
plt.figure(figsize=(8,6))
colors = ['green', 'red', 'black']
for i in range(k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Height(Inches)'], cluster_data['Weight(Pounds)'], color=colors[i], label=f'Cluster {i+1}')

# Plot Centroids
plt.scatter(centroids[:, 0],centroids[:, 1],color='purple',marker='*', s=200,label='Centroids')






