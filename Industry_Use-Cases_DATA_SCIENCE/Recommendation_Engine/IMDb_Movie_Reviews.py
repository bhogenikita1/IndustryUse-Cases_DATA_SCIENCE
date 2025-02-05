
##### 15/10/2024  #####

'''
1. You are given a dataset of movies with various attributes like genres, 
keywords, and descriptions. Your task is to build a content-based 
recommendation engine that recommends movies similar to a given movie 
based on these attributes.
Steps:
 Preprocess the Data: Extract relevant features (e.g., genres, overview).
 Vectorize the Text Data: Use TF-IDF on the overview field.
 Compute Similarity: Use cosine similarity to find similar movies.
 Recommend: Given a movie, recommend the top 10 most similar movies based on 
content.
Note: Use IMDB dataset
'''
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the dataset 
movies = pd.read_csv("C:/Users/HP/Downloads/IMDb_Movie_Reviews.csv")
movies
# Select relevant columns
movies = movies[['Review_Text']]
# Fill NaN values in overview with empty strings
movies['Review_Text'] = movies['Review_Text'].fillna('')
# Preview the dataset
movies.head()
# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
# Apply TF-IDF on the 'Review_Text' field
tfidf_matrix = tfidf.fit_transform(movies['Review_Text'])
# Print the shape of the TF-IDF matrix
print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

# Compute the cosine similarity matrix from the TF-IDF matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim
# Print the shape of the similarity matrix
print(f"Cosine Similarity Matrix Shape: {cosine_sim.shape}")

# Create a function to recommend movies based on cosine similarity
def recommend_movies(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = movies[movies['Review_Text'] == title].index[1]

    # Get the similarity scores for all movies with the given movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on similarity scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar movies (excluding the first one, which is itself)
    sim_scores = sim_scores[1:11]

    # Get the indices of these movies
    movie_indices = [i[1] for i in sim_scores]

    # Return the titles of the top 10 similar movies
    return movies['Review_Text'].iloc[movie_indices]

# Test the recommendation system
print(recommend_movies('The Dark Knight'))





'''
Q.2) Build an item-based collaborative filtering recommendation engine. 
Instead of recommending items based on similar users, recommend items 
that are similar to those that a user has already interacted with.
Steps:
 Preprocess the Data: Create a user-item matrix where rows are users and columns are 
items (movies).
 Compute Item Similarity: Calculate similarity between items based on user 
interactions.
 Recommend Items: For a given user, recommend items that are similar to those the 
user has already rated highly
'''

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
file_path="C:/Users/HP/Downloads/IMDb_Movie_Reviews.csv"
data=pd.read_csv(file_path)
data
def recommend_items(user_id, user_item_matrix, item_similarity_df, top_n=5):
    # Get the user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    
    # Multiply the user's ratings with the item similarity matrix
    user_similarities = np.dot(item_similarity_df.values, user_ratings.values)
    
    # Convert to a DataFrame for easier handling
    similar_items_df = pd.DataFrame(user_similarities, index=item_similarity_df.index, columns=['similarity_score'])
    
    # Remove items the user has already rated
    similar_items_df = similar_items_df[user_ratings == 0]
    
    # Sort by similarity score
    recommendations = similar_items_df.sort_values(by='similarity_score', ascending=False).head(top_n)
    
    return recommendations

# Example usage:
user_id = 1  # specify the user ID you want to make recommendations for
recommend_items(user_id, user_item_matrix, item_similarity_df, top_n=5)




'''
3. Using the mlxtend library, write a Python program to generate association 
rules from a dataset of transactions. The program should allow setting a 
minimum support threshold and minimum confidence threshold for rule 
generation.
transactions = [['Tea', 'Bun'], ['Tea', 'Bread'], ['Bread', 'Bun'], ['Tea', 'Bun', 
'Bread']
====>
'''

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

transactions = [['Tea', 'Bun'], ['Tea', 'Bread'], ['Bread', 'Bun'], ['Tea', 'Bun', 'Bread']]
#Transactions Encoding
te=TransactionEncoder()
transformed_data=te.fit(transactions).transform(transactions)
df=pd.DataFrame(transformed_data,columns=te.columns_)
#apriori Algorithm
frequent_itemsets=apriori(df,min_support=0.5,use_colnames=True)
rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1)
# Output the results
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])



'''
Q.4) Build a popularity-based recommendation system. The system should 
recommend movies based on their overall popularity (e.g., number of ratings 
or average rating).
Steps:
 Preprocess the Data: Calculate the total number of ratings and average rating for each 
movie.
 Rank the Movies: Rank movies based on the chosen popularity metric.
 Recommend Movies: Recommend the top N most popular movies to any user.
===>
'''
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
file_path="C:/Users/HP/Downloads/IMDb_Movie_Reviews.csv"
data=pd.read_csv(file_path)
data
scaler=MinMaxScaler()
data['Normalized_reviews']=scaler.fit_transform(data[['Review_Text']])

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
transactions=[['Apple','Banana'],['Apple','Orange'],['Banana','Orange'],['Apple','Banana','Orange']]
te=TransactionEncoder()
transformed_data=te.fit(transactions).transform(transactions)
df=pd.DataFrame(transformed_data,columns=te.columns_)
#FP-Growth Algorithm
frequent_itemsets=fpgrowth(df,min_support=0.5,use_colnames=True)
print(frequent_itemsets)






