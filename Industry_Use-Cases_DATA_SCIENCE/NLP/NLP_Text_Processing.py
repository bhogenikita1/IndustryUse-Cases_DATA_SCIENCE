# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:03:21 2024

@author: HP
"""

'''
1. Implement a simple Named Entity Recognition (NER) 
function that identifies named entities in a sentence. The 
function should return a list of these named entities.
For example, given the sentence "Ramesh lives in 
Mumbai", the function should return ["Ramesh", 
"Mumbai"].
Text1: “James is the author of Atomic Habits”
Text2: “Aarti works at Accenture”
'''

#Text2:"Aarti works at Accenture"
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

# Download required NLTK resources
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download('averaged_perceptron_tagger')

def extract_named_entities(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Get Part-of-Speech (POS) tags
    pos_tags = pos_tag(tokens)
    # Perform Named Entity Chunking
    named_entities_tree = ne_chunk(pos_tags, binary=True)
    # Extract named entities
    named_entities = [' '.join(word for word, tag in subtree.leaves())
                       for subtree in named_entities_tree
                       if isinstance(subtree, nltk.Tree)]
    return named_entities

# Example usage
Text1 = "James is the author of Atomic Habits"
Text2 = "Aarti works at Accenture"

print("Named Entities in Text1:", extract_named_entities(Text1))
print("Named Entities in Text2:", extract_named_entities(Text2))



'''
”
 2.Design a function that classifies a text as either "spam" 
or "imp" (non-spam) based on the presence of certain 
keywords. For example, if the text contains words like 
"buy", "free", "offer", or "click", it should be classified as 
"spam". If these words are not present, the text should be 
classified as "imp". The function should return the 
appropriate classification.

Text1: “Buy 1 Get 1 Free”
Text2: “Meeting is scheduled at 12 PM ”
Text2: “Click on the link below to see the offer.”
'''

def classify_text_as_spam_or_imp(text):
    # Define spam keywords
    spam_keywords = ["buy", "free", "offer", "click"]
    # Convert the text to lowercase and check for the presence of any spam keywords
    for keyword in spam_keywords:
        if keyword in text.lower():
            return "spam"
    return "imp"

# Example usage
Text1 = "Buy 1 Get 1 Free"
Text2 = "Meeting is scheduled at 12 PM"
Text3 = "Click on the link below to see the offer."

print("Text1:", classify_text_as_spam_or_imp(Text1))  # Expected: spam
print("Text2:", classify_text_as_spam_or_imp(Text2))  # Expected: imp
print("Text3:", classify_text_as_spam_or_imp(Text3))  # Expected: spam



'''
3. Create a function that should return a list of stemmed 
words.
e.g [‘running’] = [‘run’]
list = [‘painful’,’standing’,’abstraction’,’magically’]
'''

stemmer=nltk.stem.PorterStemmer()
stemmer.stem("painful")
stemmer.stem("standing")
stemmer.stem("abstraction")
stemmer.stem("magically")



'''
4. Implement a function that takes a list of tokens (words) 
and removes all stopwords from it. For example, if the 
input tokens are ["This", "is", "a", "test"] and 
the stopwords are ["is", "a", "the"], the function should 
return ["This", "test"].
Stopwords = [“is”,”a”,”the”, “an”,”she”]
Sentence1: “an apple is on the table.”
Sentence2: “She is an engineer.”
'''

import nltk
from nltk.corpus import stopwords
Sentence1= "an apple is on the table."
Sentence2= "She is an engineer."

def remove_stopwords(tokens, stopwords):
    # Convert all tokens to lowercase to handle case-insensitivity
    return [word for word in tokens if word.lower() not in stopwords]

# Example usage:
stopwords = ["is", "a", "the", "an", "she"]

# Tokenized sentences
sentence1_tokens = ["an", "apple", "is", "on", "the", "table"]
sentence2_tokens = ["She", "is", "an", "engineer"]

# Remove stopwords
filtered_sentence1 = remove_stopwords(sentence1_tokens, stopwords)
filtered_sentence2 = remove_stopwords(sentence2_tokens, stopwords)

print("Filtered Sentence 1:", filtered_sentence1)
print("Filtered Sentence 2:", filtered_sentence2)


'''
5 . Perform lemmatzation on the given text
 text= "Dancing is an art. Students should be taught 
dance as a subject in schools . I danced in many of my 
school function. Some people are always hesitating 
to dance."

lemmatization:-Contextual Analysis: 
                Part of Speech
                Valid Words:The output of lemmatization is always a valid word in the language.
Example:
The word "running" is lemmatized to "run."
The word "better" is lemmatized to "good."

libraries for lemmatization: NLTK,spaCy,TextBlob
'''
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk import word_tokenize
text = "Dancing is an art. Students should be taught dance as a subject in schools. I danced in many of my school functions. Some people are always hesitating to dance."

tokens = word_tokenize(text)
tokens

lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

lemmatized_text = ' '.join(lemmatized_words)
lemmatized_text











