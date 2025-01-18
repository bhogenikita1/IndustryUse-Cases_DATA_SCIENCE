# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:30:30 2024

@author: HP
"""

# -*- coding: utf-8 -*-

'''
Kitabi Duniya, a famous book store in India, which was established before Independence, 
the growth of the company was incremental year by year, but due to online selling of books
and wide spread Internet access its annual growth started to collapse, seeing sharp downfalls, 
you as a Data Scientist help this heritage book store gain its popularity back and increase 
footfall of customers and provide ways the business can improve exponentially, apply Association 
RuleAlgorithm, explain the rules, and visualize the graphs for clear understanding of solution.
1.) Books.csv

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

'''
The Departmental Store, has gathered the data of the products it sells on a Daily basis.
Using Association Rules concepts, provide the insights on the rules and the plots.
2.) Groceries.csv
'''

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

#Sample dataset
data1 = pd.read_csv("C:/PVD_PCA-9/groceries.csv.xls", skiprows=5, nrows=1)
print(data1)

transaction_list = data1.stack().groupby(level=0).apply(list).tolist()
#step 1: Convert the dataset into a format suitable for Apriori
te = TransactionEncoder()
te_ary= te.fit(transaction_list).transform(transaction_list)
df= pd.DataFrame(te_ary, columns=te.columns_)

#step 2: Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets= apriori(df, min_support=0.05, use_colnames=True)

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

# Step 6: Sort the rules by lift and show the top 10 rules
rules_sorted = rules.sort_values(by='lift', ascending=False)
print("\nTop 10 Association Rules by Lift:")
print(rules_sorted.head(10))

# Step 7: Visualize the top 10 association rules by Lift using a bar plot
plt.figure(figsize=(10, 6))
plt.barh(rules_sorted['antecedents'].head(10).astype(str), rules_sorted['lift'].head(10), color='skyblue')
plt.xlabel('Lift')
plt.title('Top 10 Association Rules by Lift')
plt.gca().invert_yaxis()
plt.show()

# Step 8: Visualize support vs confidence for the top 10 rules
plt.scatter(rules_sorted['support'].head(10), rules_sorted['confidence'].head(10), alpha=0.7, marker='o')
plt.title('Support vs Confidence for Top 10 Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()
#########################################################################################
'''
A film distribution company wants to target audience based on their likes and dislikes, 
you as a Chief Data Scientist Analyze the data and come up with different rules of movie list 
so that the business objective is achieved.
3.) my_movies.csv
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
'''
A Mobile Phone manufacturing company wants to launch its three brand new phone into 
the market, but before going with its traditional marketing approach this time it want 
to analyze the data of its previous model sales in different regions and you have been 
hired as an Data Scientist to help them out, use the Association rules concept and provide 
your insights to the company’s marketing team to improve its sales.
4.) myphonedata.csv
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
'''
A retail store in India, has its transaction data, and it would like to know the buying pattern of the 
consumers in its locality, you have been assigned this task to provide the manager with rules 
on how the placement of products needs to be there in shelves so that it can improve the buying
patterns of consumes and increase customer footfall. 
5.) transaction_retail.csv
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
