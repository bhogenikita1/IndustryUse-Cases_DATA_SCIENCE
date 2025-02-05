# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:28:02 2024

@author: HP
"""
'''
1.Extract reviews for any movie from IMDB and perform 
sentiment analysis.

Business Objective:
    Extract and analyze movie reviews to determine sentiment (positive, negative, or neutral).

Constraints:
    Must comply with website policies.
    Reviews may contain noise and mixed sentiments.
    Accuracy may be affected by sarcasm and short text limitations.
'''

import bs4
from bs4 import BeautifulSoup as bs
import requests
link='https://www.rottentomatoes.com/m/iron_man_3/reviews'
page=requests.get(link)
page
page.content # retrieves the raw HTML content of a web page
soup=bs(page.content,'html.parser') 

#A parser is a component in software that processes input data,
# typically written in a structured format like HTML, XML, JSON, or programming languages, 
#and breaks it down into a format that can be used by a program.

print(soup.prettify())#The prettify() function of the BeautifulSoup object structures the HTML with proper indentation and line breaks, making it easier for humans to read.



## now let us scrap review body
reviews=soup.find_all('div',class_='review-text-container')
reviews
review_body=[]
for i in range(0,len(reviews)):
    review_body.append(reviews[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
review_body[0].split('\n')[0]
review_body[1]
k=[]
for i in range(len(review_body)):
    k.append(review_body[i].split('\n')[0])
k[0]
###convert to csv file
import pandas as pd
df=pd.DataFrame()
df['review_body']=review_body
df
df.to_csv('C:/8-textmining/textmining/movie1_reviews.csv',index=True)

df=pd.read_csv('C:/8-textmining/textmining/movie1_reviews.csv')
df.sample(3)

#############################################

#sentiment analysis
import pandas as pd
from textblob import TextBlob
sent="This is very excellent movie"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv('C:/8-textmining/textmining/movie1_reviews.csv')
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']














