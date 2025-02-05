
'''
The Departmental Store, has gathered the data of the products it sells on a Daily basis.
Using Association Rules concepts, provide the insights on the rules and the plots.

2.) Groceries.csv

Business Objective:
A departmental store wants to analyze its daily sales data to understand consumer buying patterns. The goal is to use association rule mining to identify frequently bought product combinations. These insights will help in optimizing store layout, improving promotions, and increasing sales.

Constraints:
The dataset must be correctly preprocessed for the Apriori algorithm.
Appropriate support and lift thresholds should be selected for meaningful insights.
Visualization of association rules should be clear and interpretable.
The recommendations should be actionable, improving store layout and promotions.
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
