

'''
A film distribution company wants to target audience based on their likes and dislikes, 
you as a Chief Data Scientist Analyze the data and come up with different rules of movie list 
so that the business objective is achieved.

3.) my_movies.csv

Business Objective:
    A film distribution company wants to recommend movies to audiences based on their preferences. 
    The goal is to identify patterns in movie choices using association rule mining to create targeted 
    recommendations and improve customer engagement.

Constraints:
    The dataset must be in a format suitable for the Apriori algorithm.
    Minimum support and lift thresholds should be carefully chosen to ensure meaningful rules.
    The rules should be interpretable and actionable for marketing strategies.
    The dataset may have missing or redundant values, requiring preprocessing.
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
#Sample dataset
data2=pd.read_csv("C:/PVD_PCA-9/my_movies.csv.xls")
data2
transaction_list = data2.stack().groupby(level=0).apply(list).tolist()
#step 1: Convert the dataset into a format suitable for Apriori
te = TransactionEncoder()
te_ary= te.fit(transaction_list).transform(transaction_list)
df= pd.DataFrame(te_ary, columns=te.columns_)

#step 2: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets= apriori(df, min_support=0.5, use_colnames=True)

#step 3: Generate association rules from the frequent itemsets
if frequent_itemsets.empty:
    print("No frequent itemsets found.")
else:
    rules= association_rules(frequent_itemsets, metric="lift", min_threshold=1)

#step 4: Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])
##########################################################################################



