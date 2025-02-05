# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 09:05:02 2024

@author: HP
"""

'''
1.Write a NLTK program to omit some given stop words 
from the stopwords list.
Stopwords to omit : 'again', 'once', 'from'
'''

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Get the list of English stopwords
stop_words = set(stopwords.words('english'))
len(stop_words)     #179

# Words to omit
omit_words = {'again', 'once', 'from'}

# Remove the specified stopwords
filtered_stopwords = stop_words - omit_words

# Display the modified stopwords list
print("Stopwords after omission:")
print(filtered_stopwords)

len(filtered_stopwords)   #176



'''
2. Implement basic text preprocessing steps on a dataset, 
including tokenization, lowercasing, removing stopwords, 
punctuation, and special characters.
text = "Hello! This is a sample text. Let's tokenize it, remove stopwords and 
punctuations. Hope you all are doing well!"
'''
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "Hello! This is a sample text. Let's tokenize it, remove stopwords and punctuations. Hope you all are doing well!"

# Tokenization
tokens = word_tokenize(text)
tokens
# Convert to lowercase
tokens = [word.lower() for word in tokens]
tokens
# Remove punctuation and special characters
tokens = [re.sub(r'[^\w\s]', '', word) for word in tokens if word.isalnum()]
len(tokens)

stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]
tokens

print("Processed Tokens:", tokens)


'''
3.Perform basic sentiment analysis on text using a Bag-ofWords model. Build a classifier to predict whether a review is 
positive or negative.
Dataset : books.csv'''

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_excel("C:\DataSet\Bookreviews.xlsx")

# Display dataset
print(df.head())

# Example columns: 'Review', 'Sentiment' (Sentiment is either 'positive' or 'negative')

# Text preprocessing
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    
    # Remove punctuation and non-alphanumeric characters
    tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing to the 'Review' column
df['cleaned_review'] = df['Review'].apply(preprocess_text)

# Convert labels to binary values (positive = 1, negative = 0)
df['Sentiment'] = df['Sentiment'].map({'positive': 1, 'negative': 0})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['Sentiment'], test_size=0.2, random_state=42)

# Convert text to Bag-of-Words representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Build and train the Logistic Regression classifier
classifier = LogisticRegression()
classifier.fit(X_train_bow, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_bow)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])

# Display results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

















