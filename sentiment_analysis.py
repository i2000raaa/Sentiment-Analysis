import re
import datetime
import json
import nltk

lines = open('tweets.txt', "r", encoding='utf-8-sig').read().splitlines()
with open("cleaned_tweets.txt", 'w+', encoding="utf-8") as cleaned_tweets_file:
    for tweet in lines:
        # remove tags from tweets
        tweet = re.sub(r'#[ ]*[A-Za-zA-Яа-яё0-9]*', '', tweet)
        #remove RT
        tweet = re.sub(r'RT', '', tweet)
        # remove http* links
        tweet = re.sub(r'http\S+', '', tweet)
        # remove users
        tweet = re.sub(r'@[ ]*[^ \n]*', '', tweet)
        # remove special characters
        tweet = re.sub(r'[!.,@$%^&*()\-_=+\"№;?/`:<>{}\[\]\']', '', tweet)
        # remove digits
        tweet = re.sub(r'\d', '', tweet)
        #remove retweets
        tweet = re.sub(r'^(RT|rt)( @\w*)&[: ]', '', tweet)
        # remove all trailing whitespaces
        tweet = tweet.strip()
        if len(tweet) > 0:
            cleaned_tweets_file.write(tweet + '\n')



