# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 23:48:44 2025

@author: HP
"""

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
