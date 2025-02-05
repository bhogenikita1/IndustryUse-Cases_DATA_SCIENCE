# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:30:30 2024

@author: HP
"""



'''
Kitabi Duniya, a famous book store in India, which was established before Independence, 
the growth of the company was incremental year by year, but due to online selling of books
and wide spread Internet access its annual growth started to collapse, seeing sharp downfalls, 
you as a Data Scientist help this heritage book store gain its popularity back and increase 
footfall of customers and provide ways the business can improve exponentially, apply Association 
RuleAlgorithm, explain the rules, and visualize the graphs for clear understanding of solution.

1.) Books.csv

Business Objective:
Kitabi Duniya, a historic bookstore in India, has seen a decline in sales due to the rise of online book marketplaces and increased internet access. The goal is to analyze past sales data using association rule mining to identify customer purchase patterns. The insights will help in optimizing book placement, bundling offers, and designing targeted promotions to attract more customers and revive sales.

Constraints:
The dataset should be in a proper format for the Apriori algorithm.
Selecting the right support and lift thresholds is key to obtaining meaningful rules.
Visualization of the rules should be done for better understanding.
The strategy should align with customer preferences and reading trends.
The store may need to combine offline and online strategies for better reach.

'''

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
#Sample dataset
data=pd.read_csv("C:/PVD_PCA-9/book.csv.xls")
data
transaction_list = data.stack().groupby(level=0).apply(list).tolist()
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
    rules= association_rules(frequent_itemsets, metric="lift", min_threshold=11)

#step 4: Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents','consequents','support','confidence','lift']])
####################################################################################

