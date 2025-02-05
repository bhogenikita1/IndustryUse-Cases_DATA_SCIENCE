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


