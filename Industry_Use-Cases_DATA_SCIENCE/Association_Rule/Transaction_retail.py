

'''
A retail store in India, has its transaction data, and it would like to know the buying pattern of the 
consumers in its locality, you have been assigned this task to provide the manager with rules 
on how the placement of products needs to be there in shelves so that it can improve the buying
patterns of consumes and increase customer footfall. 
5.) transaction_retail.csv

Business Objective:
A retail store in India wants to analyze its transaction data to understand customer buying patterns. The goal is to identify frequently purchased product combinations using association rule mining. This insight will help optimize product placement on shelves to encourage more sales and improve customer footfall.

Constraints:
The dataset should be properly formatted for the Apriori algorithm.
Selecting the right support and lift thresholds is crucial for generating actionable rules.
Product placement recommendations should be practical and aligned with customer preferences.
The analysis must account for seasonality, regional trends, and potential missing data.
'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
#Sample dataset
data4=pd.read_csv("C:/PVD_PCA-9/transactions_retail1.csv")
data4
transaction_list = data4.stack().groupby(level=0).apply(list).tolist()
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
    rules= association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)

#step 4: Output the results
print("Frequent Itemsets:",frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])
