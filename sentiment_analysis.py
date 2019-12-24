import re
import datetime
import time
import json
import nltk
from pymystem3 import Mystem
import pymorphy2

start = time.time()
Tweets = {}

# Step 1. Clean raw tweets fron links, retweets, special characters, whitespaces 
lines = open('tweets.txt', "r", encoding='utf-8-sig').read().splitlines()
with open("cleaned_tweets.txt", 'w+', encoding="utf-8-sig") as cleaned_tweets_file:
    for tweet in lines:
        # get date
        tweet_date = tweet[:16]  # tweet_date = datetime.datetime.strptime(tweet_date_str, '%Y-%m-%d %H:%M').date()
        tweet = tweet [16:]
        # remove retweets
        tweet = re.sub(r'[ ](RT|rt)( @\w*)[: ]', '', tweet)
        tweet = re.sub(r'[ ](RT|rt)', '', tweet)
        # remove tags from tweets
        tweet = re.sub(r'#[ ]*[A-Za-zA-Яа-яё0-9]*', '', tweet)
        # remove http* links
        tweet = re.sub(r'http\S+', '', tweet)
        # remove users
        tweet = re.sub(r'@[ ]*[^ \n]*', '', tweet)
        # remove digits
        tweet = re.sub(r'\d', '', tweet)
        tweet = tweet.strip()
        # remove retweets
        tweet = re.sub(r'(RT|rt)( @\w*)&[: ]', '', tweet)
        # remove special characters
        tweet = re.sub(r'[!.,@$%^&*()\-_=+\"№;?/`:<>{}\[\]\']', '', tweet)
        # remove whitespaces
        tweet = tweet.strip()
        if len(tweet) > 0:
            Tweets[tweet_date] = tweet.split(' ')
        cleaned_tweets_file.write(json.dumps(Tweets))

print ('Step 1. Clean raw tweets fron links, retweets, special characters, whitespaces finished in ' + str("{0:.2f}".format(time.time() - start))+ ' seconds.') 
start = time.time()

# Step 2. Stemmeing and lemmartizing
morph = pymorphy2.MorphAnalyzer()    
with open("lemmatized_tweets.txt", 'w+', encoding="utf-8-sig") as lemmatized_tweets_file:
    for tweet_date, tweet in Tweets.items():
        for index, word in enumerate(tweet, start=0):
            tweet[index] = morph.parse(word)[0].normal_form
        lemmatized_tweets_file.write(json.dumps(Tweets))

print ('Step 2. Stemming and lemmartizing finished in ' + str("{0:.2f}".format(time.time() - start)) + ' seconds.') 
start = time.time()

# Step 3. Remove stop russian words
russian_stop_words = nltk.corpus.stopwords.words('russian')
with open("tweets_wo_stopwords.txt", 'w+', encoding="utf-8-sig") as tweets_wo_stopwords_file:
    for tweet_date, tweet in Tweets.items():
        Tweets[tweet_date] = [word for word in tweet if word not in russian_stop_words]        
    tweets_wo_stopwords_file.write(json.dumps(Tweets))

print ('Step 3. Remove stop russian words finished in ' + str("{0:.2f}".format(time.time() - start)) + ' seconds.') 
start = time.time()
