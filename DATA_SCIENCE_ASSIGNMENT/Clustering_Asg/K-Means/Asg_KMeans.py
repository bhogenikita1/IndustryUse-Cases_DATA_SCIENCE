'''
1. Perform K means clustering on the airlines dataset to obtain optimum number of clusters. 
Draw the inferences from the clusters obtained. Refer to EastWestAirlines.xlsx dataset.
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




'''
2.	Perform clustering for the crime data and identify the number of clusters formed and draw inferences. Refer to crime_data.csv dataset.
    formed and draw inferences. Refer to crime_data.csv dataset.
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



'''
3.	Analyze the information given in the following ‘Insurance Policy dataset’to             
create clusters of persons falling in the same type. 
Refer to Insurance Dataset.csv
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



'''
4.	Perform clustering analysis on the telecom dataset. 
The data is a mixture of both categorical and numerical data.
 It consists of the number of customers who churn. 
 Derive insights and get possible information on factors that may affect the churn decision. 
 Refer to Telco_customer_churn.xlsx dataset.
'''

'''
Categorical variables: Referred a Friend, Offer, Phone Service, Multiple Lines, 
                        Internet Type, Unlimited Data, Contract, etc.
Numerical variables: Tenure in Months, Avg Monthly Long Distance Charges,
                     Monthly Charge, Total Charges, etc.

#Define objectives
#Data Preparation=1]cleaning
                    2]Normalization
                    3]convert categorical data
'''                    



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df=pd.read_excel("C:/DataSet/Telco_customer_churn.xlsx")
df
print(df.head())
print(df.info())
print(df.describe())
df.columns

df.isna().sum()

df.iloc[:,[0,5]]
df.iloc[:,[6,10]]

# Calculate mean of 'Monthly Charge'
df['Monthly Charge'].mean()

# Fill missing values with the median of 'Monthly Charge'
df['Monthly Charge'].fillna(df['Monthly Charge'].median(), inplace=True)

df['Monthly Charge'].plot(kind='box')

df['Monthly Charge'].plot(kind='kde')

set(df['Monthly Charge'])

def norm_func(i):
    return (i - i.min()) / (i.max() - i.min())

# Normalize the selected columns
df_norm = norm_func(df[['Monthly Charge']])
df_norm

# Step 1: Apply K-Means Clustering
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Monthly Charge', 'Tenure in Months']])  
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
plt.scatter(df1['Monthly Charge'], df1['Tenure in Months'], color='orange',label='Cluster 0')
plt.scatter(df1['Monthly Charge'], df2['Tenure in Months'], color='blue', label='Cluster 1')
plt.scatter(df1['Monthly Charge'], df3['Tenure in Months'], color='green',label='Cluster 2')


# Plot the centroids
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], 
            color='purple', marker='*', s=200, label='Centroid')

# Set plot title and labels
plt.title('K-Means Clustering')
plt.xlabel('Monthly Charge')
plt.ylabel('Tenure in Months')
plt.legend()
plt.show()

#  preprocessing

scaler=MinMaxScaler()
scaler.fit(df[['Mothly Charge']])
df['Mothly Charge']=scaler.transform(df[['Mothly Charge']])

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




'''
5.Perform clustering on mixed data.
 Convert the categorical variables to numeric by using dummies or 
 label encoding and perform normalization techniques. 
 The dataset has the details of customers related to their auto insurance. 
 Refer to Autoinsurance.csv dataset.
'''


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


df=pd.read_csv("C:/DataSet/Insurance_Dataset.csv")
df




'''
6. You are given a dataset with two numerical features Height and Weight. 
Your goal is to cluster these people into 3 groups using K-Means clustering. 
After clustering, you will visualize the clusters and their centroids.
 Load the dataset (or generate random data for practice).
 Apply K-Means clustering with k = 3.
 Visualize the clusters and centroids.
 Experiment with different values of k and see how the clustering chang
'''
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.cluster.hierarchy import linkage

df=pd.read_csv("C:/DataSet/HeightWeight.csv")
df.head()

plt.scatter(df['Height(Inches)'],df['Weight(Pounds)'])
plt.xlabel('Height(Inches)')
plt.ylabel('Weight(Pounds)')
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Height(Inches)','Weight(Pounds)']])
y_predicted
df['cluster']=y_predicted
df.head()
km.cluster_centers_

df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1['Height(Inches)'],df1['Weight(Pounds)'],color='Green')
plt.scatter(df2['Height(Inches)'],df2['Weight(Pounds)'],color='red')
plt.scatter(df3['Height(Inches)'],df3['Weight(Pounds)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Height(Inches)')
plt.ylabel(' Weight(Pounds)')
plt.legend()



















