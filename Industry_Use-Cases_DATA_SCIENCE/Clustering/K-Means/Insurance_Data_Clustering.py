

'''
3.	Analyze the information given in the following ‘Insurance Policy dataset’to             
create clusters of persons falling in the same type. 
Refer to Insurance Dataset.csv

Business Objective:
Segment customers based on their insurance-related attributes (e.g., premiums paid, age, claims made, income).
Identify distinct customer groups to optimize insurance policy offerings and pricing strategies.
Improve customer targeting for marketing and personalized policy recommendations.

Constraints:
Proper selection of the number of clusters (Elbow Method or Silhouette Score is recommended).
Features must be scaled appropriately to ensure fair clustering results.
K-Means assumes spherical and equally sized clusters, which might not always be realistic.
Outliers or skewed data distributions may impact cluster assignments.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("C:/DataSet/Insurance_Dataset.csv")
df
# Specify the columns to be used for clustering
columns_to_cluster = ['Premiums_Paid', 'Age', 'Days_to_Renew', 'Claims_made', 'Income']

# Initial scatter plot to visualize Age vs Income for a quick view
plt.scatter(df['Age'], df['Income'])
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()


# Preprocessing: Scale the data using Min-Max scaling for consistent ranges
scaler = MinMaxScaler()

# Fit the scaler on the selected columns and transform the data
df_scaled = scaler.fit_transform(df[columns_to_cluster])

# Convert the scaled data back into a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=columns_to_cluster)

# Apply KMeans clustering with 3 clusters
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df_scaled)  # Predict cluster labels for each data point

# Add the predicted cluster labels to the original DataFrame
df['cluster'] = y_predicted

# Display the first few rows to check the clusters
df.head()

# Print the cluster centers in the scaled version
print("Cluster Centers (scaled):")
print(km.cluster_centers_)

# Inverse transform the cluster centers to the original scale (undo the scaling)
centroids_original = scaler.inverse_transform(km.cluster_centers_)
print("Cluster Centers (original scale):")
print(centroids_original)

# Separate the data into different clusters for plotting
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plot the clusters based on Age and Income
plt.scatter(df1['Age'], df1['Income'], color='orange', label='Cluster 0')
plt.scatter(df2['Age'], df2['Income'], color='blue', label='Cluster 1')
plt.scatter(df3['Age'], df3['Income'], color='green', label='Cluster 2')

# Plot the centroids of the clusters
plt.scatter(centroids_original[:, 1], centroids_original[:, 4], 
            color='purple', marker='*', s=200, label='Centroid')

# Set the title and labels for the plot
plt.title('K-Means Clustering')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()


####### ELBOW CURVE

# Scaling the data for the Elbow method to determine the optimal number of clusters
scaler = MinMaxScaler()
columns_to_cluster = ['Premiums_Paid', 'Age', 'Days_to_Renew', 'Claims_made', 'Income']


# Apply normalization to the selected columns
def norm_func(i):
    return (i - i.min()) / (i.max() - i.min())

# Normalize the selected columns
df_norm = norm_func(df[[columns_to_cluster]])

# Using the Elbow method to find the optimal number of clusters
TWSS = []  # Total Within-Cluster Sum of Square (TWSS)
k = list(range(2, 8))  # Range of clusters to try

# Fit KMeans for each k and calculate the inertia (TWSS)
for i in k:
    km = KMeans(n_clusters=i)
    km.fit(df(columns_to_cluster))
    TWSS.append(km.inertia_)

# Plot the Elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k, TWSS, 'ro-')  # 'ro-' gives red circles connected by lines
plt.xlabel('Number of clusters')
plt.ylabel('TWSS (Inertia)')
plt.title('Elbow Method For Optimal Number of Clusters')
plt.show()




##Kmeans clustering

# Importing required libraries
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Load the dataset
df = pd.read_csv("C:/DataSet/Insurance_Dataset.csv")
df.head()
df.columns
# Select relevant columns for clustering
columns_to_cluster = ['Premiums_Paid', 'Age', 'Days_to_Renew', 'Claims_made', 'Income']

# Initial scatter plot (e.g., Age vs Income for quick visualization)
plt.scatter(df['Age'], df['Income'])
plt.xlabel('Age')
plt.ylabel('Income')
plt.show()

# Preprocessing using Min-Max Scaler
scaler = MinMaxScaler()

# Fit and transform the selected columns
df_scaled = scaler.fit_transform(df[columns_to_cluster])

# Convert the scaled data back to a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=columns_to_cluster)

# Initialize KMeans
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df_scaled)

# Add the cluster labels to the original dataframe
df['cluster'] = y_predicted

# Display the first few rows of the updated dataframe
print(df.head())

# Display cluster centers in the scaled space
print("Cluster Centers (scaled):")
print(km.cluster_centers_)

# Inverse transform the cluster centers back to the original scale
centroids_original = scaler.inverse_transform(km.cluster_centers_)
print("Cluster Centers (original scale):")
print(centroids_original)

# Creating dataframes for each cluster
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

# Plotting each cluster with different colors (Age vs Income as an example)
plt.scatter(df1['Age'], df1['Income'], color='green', label='Cluster 1')
plt.scatter(df2['Age'], df2['Income'], color='red', label='Cluster 2')
plt.scatter(df3['Age'], df3['Income'], color='black', label='Cluster 3')

# Plotting the cluster centers
plt.scatter(centroids_original[:, 1], centroids_original[:, 4], color='purple', marker='*', s=200, label='Centroid')
plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.show()

