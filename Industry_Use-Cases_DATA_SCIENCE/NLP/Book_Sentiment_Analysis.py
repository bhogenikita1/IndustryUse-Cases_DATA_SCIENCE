# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:48:58 2025

@author: HP
"""


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

