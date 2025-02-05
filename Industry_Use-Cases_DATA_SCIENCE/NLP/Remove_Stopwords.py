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




















