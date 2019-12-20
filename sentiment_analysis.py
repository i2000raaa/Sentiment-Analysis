import re
import datetime
import json
import nltk
from pymystem3 import Mystem

start = time.time()

# Step 1. Clean raw tweets fron links, retweets, special characters, whitespaces 
lines = open('tweets.txt', "r", encoding='utf-8-sig').read().splitlines()
with open("cleaned_tweets.txt", 'w+', encoding="utf-8") as cleaned_tweets_file:
    for tweet in lines:
        # remove retweets
        tweet = re.sub(r'[ ](RT|rt)( @\w*)[: ]', '', tweet)
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

print ('Step 1. Clean raw tweets fron links, retweets, special characters, whitespaces finished in ' + str("{0:.2f}".format(time.time() - start))+ ' seconds.') 
start = time.time()

# Step 2. Stemmeing and lemmartizing
cleaned_tweets = open('cleaned_tweets.txt', "r", encoding='utf-8-sig').read().splitlines()
with open("lemmatized_tweets.txt", 'w+', encoding="utf-8") as lemmatized_tweets_file:
    m = Mystem()
    for tweet in cleaned_tweets:
        lemmatized_tweet = m.lemmatize(tweet)
        s = ', '
        lemmatized_tweets_file.write(s.join(lemmatized_tweet) + '\n')

print ('Step 2. Stemmeing and lemmartizing finished in ' + str("{0:.2f}".format(time.time() - start)) + ' seconds.') 
start = time.time()

# Step 3. Remove stop russian words
russian_stop_words = nltk.corpus.stopwords.words('russian')
lemmatized_tweets = open('lemmatized_tweets.txt', "r", encoding='utf-8-sig').read().splitlines()
with open("tweets_wo_stopwords.txt", 'w+', encoding="utf-8") as tweets_wo_stopwords_file:
    for tweet in lemmatized_tweets:
        tw = tweet.split(', ')
        tweet_wo_stopwords = [word for word in tw if word not in russian_stop_words]
        tweets_wo_stopwords_file.write(s.join(tweet_wo_stopwords) + '\n')

print ('Step 3.Remove stop russian words finished in ' + str("{0:.2f}".format(time.time() - start)) + ' seconds.') 
start = time.time()
