import re
import datetime
import json
import nltk
from pymystem3 import Mystem

# Step 1. Clean raw tweets fron links, retweets, special characters, whitespaces 
lines = open('tweets.txt', "r", encoding='utf-8-sig').read().splitlines()
with open("cleaned_tweets.txt", 'w+', encoding="utf-8") as cleaned_tweets_file:
    for tweet in lines:
        print(tweet)
        # remove retweets
        tweet = re.sub(r'[ ](RT|rt)( @\w*)[: ]', '', tweet)
        print(tweet)
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
            cleaned_tweets_file.write(tweet + '\n')

# Step 2. Stemm and lemmartize tweets
cleaned_tweets = open('cleaned_tweets.txt', "r", encoding='utf-8-sig').read().splitlines()
with open("lemmatized_tweets.txt", 'w+', encoding="utf-8") as lemmatized_tweets_file:
m = Mystem()
for tweet in cleaned_tweets:
    lemmatized_tweet = m.lemmatize(tweet)
    lemmatized_tweets_file.write(lemmatized_tweet + '\n')
    print(lemmatized_tweet)

