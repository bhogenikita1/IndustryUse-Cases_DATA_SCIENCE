

'''
A Mobile Phone manufacturing company wants to launch its three brand new phone into 
the market, but before going with its traditional marketing approach this time it want 
to analyze the data of its previous model sales in different regions and you have been 
hired as an Data Scientist to help them out, use the Association rules concept and provide 
your insights to the company’s marketing team to improve its sales.
4.) myphonedata.csv

Business Objective:
    A mobile phone manufacturing company plans to launch three new models and wants to 
    improve its marketing strategy by analyzing past sales data across different regions. 
    The goal is to use association rule mining to identify patterns in customer purchases and 
    optimize product bundling, pricing, and targeted promotions.


Constraints:
    The dataset must be preprocessed properly for the Apriori algorithm.
    Choosing appropriate support and lift thresholds is crucial to extracting meaningful insights.
    The company must ensure that recommendations align with business goals and regional preferences.
    Any missing or redundant values in the dataset should be handled before analysis.

'''
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
#Sample dataset
data3=pd.read_csv("C:/PVD_PCA-9/myphonedata.csv.xls")
data3
transaction_list = data3.stack().groupby(level=0).apply(list).tolist()
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

####################################################################################

