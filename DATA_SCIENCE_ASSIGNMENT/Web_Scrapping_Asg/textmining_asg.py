# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:28:02 2024

@author: HP
"""
'''
2.Extract reviews for any movie from IMDB and perform 
sentiment analysis.
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

''''
1.Extract reviews of any product from e-commerce website Amazon.
2.Perform sentiment analysis on this extracted data and build a 
unigram and bigram word cloud. 
'''

import bs4
from bs4 import BeautifulSoup as bs
import requests
link='https://www.amazon.in/Zebronics-ZEB-K20-USB-Keyboard-Rupee/dp/B07KX9V8J4/ref=sr_1_4?crid=2UA6KEMKN65I8&dib=eyJ2IjoiMSJ9.FxG2iFdMGtFGHRuKLcCmmYjsSDY4zh8wvFsawhKCSHc_tGlFImj4f3M-mnZf8QWTnwcSMCFoC7vWPhIvIxc_8Xi6RbtJ8-OA8j33Sh89UNZ_krxumZJHzJBL-axdA1SbWx6obP-4G416Qp_GizeZMnXrWD31kkvhUF0ErPBzw1gL3YbnLndduOzbuY448mCgj6VJVL1O7FfiW66PpFeSxChcFQB6XzjIiIdCxY_AhYw.wHzI7txywJwvzCOhbqFIF2ux3Jwy6kWmuyv_gjQ1-2g&dib_tag=se&keywords=keyboard&qid=1727693760&sprefix=keyboar%2Caps%2C308&sr=8-4#customerReviews'
page=requests.get(link)
page
page.content

soup=bs(page.content,'html.parser')
soup

print(soup.prettify())

title1=soup.find_all('a',class_='a-size-base a-link-normal review-title a-color-base review-title-content a-text-bold')
title1
  
review_titles1=[]
for i in range(0,len(title1)):
    review_titles1.append(title1[i].get_text().split('\n')[1])
review_titles1  

review_titles1=[ title1.strip('\n')for title1 in review_titles1]
review_titles1
len(review_titles1)    
     
####################
    
rating1=soup.find_all('i',class_='a-icon a-icon-star a-star-5 review-rating')
rating1

rate1=[]
for i in range(0,len(rating1)):
    rate1.append(rating1[i].get_text())

rate1=[ rating1.strip('\n') for rating1 in rate1]
rate1
rate2=[float(rating1.split(' ')[0]) for rating1 in rate1]
rate2
len(rate2)

####################

reviews1=soup.find_all('div',class_='a-expander-content reviewText review-text-content a-expander-partial-collapse-content')
reviews1
review_body1=[]
for i in range(0,len(reviews1)):
    review_body1.append(reviews1[i].get_text())
review_body1
review_body1=[ reviews1.strip('\n\n')for reviews1 in review_body1]
len(review_body1)

####################

#Creating csv file
import pandas as pd
df=pd.DataFrame()
df['title1']=review_titles1
#df['rating1']=rate2
df['review_body1']=review_body1
df
df.to_csv('C:/8-textmining/textmining/keyboard1_reviews.csv',index=True)

df=pd.read_csv('C:/8-textmining/textmining/keyboard1_reviews.csv')
df.head()

#Sentiment analysis
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
sent="This is very excellent keyboard"
pol=TextBlob(sent).sentiment.polarity
pol
df=pd.read_csv('C:/8-textmining/textmining/keyboard1_reviews.csv')
df.head()
df['polarity']=df['review_body1'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']


#######################

'''
3.Choose any other website on the internet and do some research on how to extract 
text and perform sentiment analysis
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








