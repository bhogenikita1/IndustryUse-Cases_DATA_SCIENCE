

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


