
'''
3.Choose any other website on the internet and do some research on how to extract 
text and perform sentiment analysis

Business Objective:
To extract and analyze product reviews from Snapdeal (or any other e-commerce website) to understand customer sentiments. This can help businesses improve product quality, customer satisfaction, and marketing strategies.

Constraints:
Website Restrictions: Some websites block scraping; ensure compliance with robots.txt.
Data Availability: Reviews and ratings may not always be structured uniformly.
Text Processing Challenges: Reviews may contain noise, slang, or multilingual content.
Sentiment Accuracy: Sentiment analysis using TextBlob may have limitations in understanding context and sarcasm.
Scalability: The script should efficiently handle large amounts of reviews.

'''

import bs4
from bs4 import BeautifulSoup as bs
import requests
link='https://www.snapdeal.com/product/sitanjali-yellow-georgette-saree-single/637753106740'
page=requests.get(link)
page

page.content 
soup=bs(page.content,'html.parser') 

print(soup.prettify())

## now let us scrap title
title=soup.find_all('div',class_='head')
title
k
review_titles=[]
for i in range(0,len(title)):
    review_titles.append(title[i].get_text())
review_titles    

review_titles=[ title.strip('\n')for title in review_titles]
review_titles
len(review_titles)

## now let us scrap rating
rating=soup.find_all('i',class_='rating')
rating

rate=[]
for i in range(0,len(rating)):
    rate.append(rating[i].get_text())
rate
rate=[ rating.strip('\n') for rating in rate]

rate2=[]
for i in range(len(rate)):
    rate2.append(int(rate[i].split('/')[0]))
rate2

## now let us scrap review body
reviews=soup.find_all('div',class_='LTgray grey-div hf-review')
reviews
review_body=[]
for i in range(0,len(reviews)):
    review_body.append(reviews[i].get_text())
review_body
review_body=[ reviews.strip('\n\n')for reviews in review_body]
review_body[0]

## convert to csv file
import pandas as pd
df=pd.DataFrame()
df['title']=review_titles
df['rating']=rate2
df['review_body']=review_body
df

df.to_csv('C:/8-textmining/textmining/saree_reviews.csv',index=True)

df=pd.read_csv('C:/8-textmining/textmining/saree_reviews.csv')
df.sample(3)

############################

## sentiment analysis
import pandas as pd
from textblob import TextBlob
sent="This is very excellent saree"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv('C:/8-textmining/textmining/saree_reviews.csv')
df.head()
df['polarity']=df['review_body'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']



import re
from nltk.util import ngrams
 
s = "Natural-language processing (NLP) is an area of computer science " \
    "and artificial intelligence concerned with the interactions " \
    "between computers and human (natural) languages."
 
s = s.lower()
s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
tokens = [token for token in s.split(" ") if token != ""]
output = list(ngrams(tokens, 1))
output


import bs4
from bs4 import BeautifulSoup as bs
import requests
link='https://sanjivani.edupluscampus.com/'
page=requests.get(link)
page
page.content # retrieves the raw HTML content of a web page
soup=bs(page.content,'html.parser') 

