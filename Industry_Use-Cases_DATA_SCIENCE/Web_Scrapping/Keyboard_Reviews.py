
''''
1.Extract reviews of any product from e-commerce website Amazon.
2.Perform sentiment analysis on this extracted data and build a 
unigram and bigram word cloud. 

Business Objective:
Extract and analyze Amazon product reviews to understand customer sentiment and visualize insights using word clouds.

Constraints:
Scraping Restrictions: Amazon blocks scrapers; alternative methods may be needed.
HTML Structure Changes: Affect extraction accuracy.
Limited Data: Some products may have few reviews.
Sentiment Accuracy: TextBlob may misinterpret context.
Performance Issues: Large datasets may slow processing.
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



